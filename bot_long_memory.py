"""
Telegram-бот с долгой памятью (RAG — Retrieval-Augmented Generation).

Пользователь загружает документ (PDF / TXT / DOCX), бот извлекает текст,
разбивает его на чанки, строит эмбеддинги и сохраняет их в локальную
векторную базу ChromaDB. На вопросы бот отвечает строго по найденным
фрагментам документа через ProxyAPI (AI Embeddings + ChatCompletion).
"""

import asyncio
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Tuple

import aiohttp
import chromadb
from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

# ─── Загрузка переменных окружения ───────────────────────────────────────────

load_dotenv()

# ─── Константы ───────────────────────────────────────────────────────────────

CHAT_MODEL: str = "openai/gpt-5.6-luna"                  # Модель для ответов
EMBEDDING_MODEL: str = "openai/text-embedding-3-small"   # Модель для эмбеддингов
CHAT_URL: str = "https://api.proxyapi.ru/v1/chat/completions"
EMBEDDINGS_URL: str = "https://api.proxyapi.ru/v1/embeddings"

MEMORY_DIR: str = "./memory"     # Persistent storage ChromaDB
CHUNK_SIZE: int = 500            # Размер чанка в символах
CHUNK_OVERLAP: int = 50          # Перекрытие соседних чанков
TOP_K: int = 3                   # Сколько фрагментов искать на вопрос
MAX_RETRIES: int = 3             # Повторные попытки при ошибке API
RETRY_DELAY: int = 2             # Задержка между повторами (сек)

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".pdf", ".txt", ".docx")

# ─── Логирование ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Инициализация ChromaDB ──────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path=MEMORY_DIR)
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

# ─── Роутер ──────────────────────────────────────────────────────────────────

router = Router()


# ─── 1. Чтение документов ────────────────────────────────────────────────────


def load_document(file_path: str) -> str:
    """Прочитать документ и вернуть его текст.

    Поддерживаются форматы PDF, TXT и DOCX.

    Args:
        file_path: Путь к файлу на диске.

    Returns:
        Извлечённый текст документа.

    Raises:
        ValueError: Если формат файла не поддерживается.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if ext == ".docx":
        doc = DocxDocument(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)

    raise ValueError(f"Неподдерживаемый формат: {ext}")


# ─── 2. Разбиение на чанки ───────────────────────────────────────────────────


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Разбить текст на фрагменты размером ~chunk_size символов.

    Границы чанков стараемся ставить на конце предложения. Соседние чанки
    перекрываются на `overlap` символов, чтобы не терять смысл на границах.

    Args:
        text: Исходный текст.
        chunk_size: Желаемый размер чанка в символах.
        overlap: Перекрытие между соседними чанками.

    Returns:
        Список непустых текстовых чанков.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap должен быть меньше chunk_size")

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        piece = text[start:end]

        # Двигаем границу назад к концу предложения, если мы не в конце текста
        if end < length:
            boundary = max(piece.rfind(". "), piece.rfind("! "), piece.rfind("? "))
            if boundary > chunk_size // 2:
                end = start + boundary + 1
                piece = text[start:end]

        chunks.append(piece.strip())

        if end >= length:
            break

        # Следующий чанок начинается с перекрытием
        start = max(end - overlap, start + 1)

    return [chunk for chunk in chunks if chunk]


# ─── 3. Эмбеддинги ───────────────────────────────────────────────────────────


async def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Получить векторы эмбеддингов для списка текстов через ProxyAPI.

    Args:
        chunks: Список текстовых фрагментов.

    Returns:
        Список эмбеддингов (по одному вектору на каждый чанк).

    Raises:
        RuntimeError: Если API недоступен после MAX_RETRIES попыток.
    """
    api_key = os.environ["PROXYAPI_API_KEY"]
    payload: Dict[str, Any] = {"model": EMBEDDING_MODEL, "input": chunks}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    EMBEDDINGS_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"Embeddings API error {resp.status}: {body}")

                    data = await resp.json()

            items = sorted(data["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in items]

        except (aiohttp.ClientError, RuntimeError, KeyError) as exc:
            last_error = exc
            logger.warning("Embedding attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)

    raise RuntimeError(f"Не удалось получить эмбеддинги после {MAX_RETRIES} попыток: {last_error}")


# ─── 4. Поиск контекста ──────────────────────────────────────────────────────


async def retrieve_context(question: str, top_k: int = TOP_K) -> List[str]:
    """Найти релевантные фрагменты документов по вопросу.

    Эмбеддинг вопроса сравнивается с сохранёнными векторами по косинусному
    расстоянию, возвращаются `top_k` ближайших чанков.

    Args:
        question: Текст вопроса пользователя.
        top_k: Количество возвращаемых фрагментов.

    Returns:
        Список текстов релевантных чанков (от наиболее к наименее похожему).
    """
    query_embeddings = await embed_chunks([question])

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=min(top_k, collection.count()),
    )

    documents: List[str] = results.get("documents") or [[]]
    return [doc for doc in documents[0] if doc]


# ─── 5. Генерация ответа ─────────────────────────────────────────────────────


async def answer_question(question: str, contexts: List[str]) -> str:
    """Сгенерировать ответ на вопрос по найденному контексту.

    Модель инструктируется отвечать только на основе фрагментов документа,
    чтобы исключить выдумывание.

    Args:
        question: Вопрос пользователя.
        contexts: Релевантные фрагменты документа.

    Returns:
        Текст ответа модели.

    Raises:
        RuntimeError: Если API недоступен после MAX_RETRIES попыток.
    """
    context_block = "\n\n".join(
        f"[Фрагмент {i + 1}]\n{chunk}" for i, chunk in enumerate(contexts)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — ассистент, который отвечает на вопросы строго по предоставленным "
                "фрагментам документа. Не выдумывай факты. Если ответа в фрагментах нет, "
                "так и скажи: 'В документе нет информации для ответа на этот вопрос.' "
                "Отвечай на языке вопроса."
            ),
        },
        {
            "role": "user",
            "content": f"Фрагменты документа:\n{context_block}\n\nВопрос: {question}",
        },
    ]

    api_key = os.environ["PROXYAPI_API_KEY"]
    payload: Dict[str, Any] = {"model": CHAT_MODEL, "messages": messages}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    CHAT_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"Chat API error {resp.status}: {body}")

                    data = await resp.json()

            return data["choices"][0]["message"]["content"]

        except (aiohttp.ClientError, RuntimeError, KeyError, IndexError) as exc:
            last_error = exc
            logger.warning("Chat attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)

    raise RuntimeError(f"Не удалось получить ответ после {MAX_RETRIES} попыток: {last_error}")


# ─── Вспомогательная отправка ответа ─────────────────────────────────────────


async def _safe_send(bot: Bot, chat_id: int, text: str) -> None:
    """Отправить ответ, разбив его на части, если он длиннее лимита Telegram."""
    limit = 4096
    parts = [text[i : i + limit] for i in range(0, len(text), limit)] or ["(пустой ответ)"]

    for part in parts:
        try:
            await bot.send_message(chat_id=chat_id, text=part)
        except (TelegramRetryAfter, TelegramAPIError) as exc:
            logger.error("Failed to send message to %d: %s", chat_id, exc)


# ─── Обработчики Telegram ────────────────────────────────────────────────────


@router.message(Command("start"))
async def cmd_start(message: Message, bot: Bot) -> None:
    """Показать приветствие и инструкцию."""
    await bot.send_message(
        message.chat.id,
        "Привет! Я бот с долгой памятью на базе RAG.\n\n"
        "1. Пришли мне документ (PDF, TXT или DOCX) — я разобью его на фрагменты "
        "и сохраню в векторную базу ChromaDB.\n"
        "2. Затем задавай вопросы — я отвечу только по содержимому документа.\n\n"
        "Команды:\n"
        "/clear — удалить все документы из базы",
    )


@router.message(Command("clear"))
async def cmd_clear(message: Message, bot: Bot) -> None:
    """Очистить векторную базу от всех сохранённых документов."""
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
        await bot.send_message(
            message.chat.id,
            f"База очищена: удалено фрагментов — {len(all_ids)}.",
        )
    else:
        await bot.send_message(message.chat.id, "В базе уже пусто.")


@router.message(F.document)
async def handle_document(message: Message, bot: Bot) -> None:
    """Принять документ: сохранить файл, извлечь текст, построить эмбеддинги."""
    document = message.document
    file_name = document.file_name or "document.bin"
    ext = os.path.splitext(file_name)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        await bot.send_message(
            message.chat.id,
            f"Неподдерживаемый формат '{ext or 'без расширения'}'. "
            f"Пришли файл: {', '.join(SUPPORTED_EXTENSIONS)}.",
        )
        return

    temp_path = f"temp_{uuid.uuid4().hex}{ext}"

    try:
        await bot.send_message(message.chat.id, f"Загружаю «{file_name}»...")

        # Скачиваем файл из Telegram во временный путь
        file_info = await bot.get_file(document.file_id)
        await bot.download_file(file_info.file_path, temp_path)

        # Извлекаем и разбиваем текст
        text = load_document(temp_path)
        if not text.strip():
            await bot.send_message(
                message.chat.id,
                "В документе не удалось найти текст (возможно, это скан без распознавания).",
            )
            return

        chunks = chunk_text(text)
        await bot.send_message(
            message.chat.id,
            f"Текста извлечено: {len(text)} символов. Фрагментов: {len(chunks)}. Строю эмбеддинги...",
        )

        # Эмбеддинги + сохранение в ChromaDB батчами
        embeddings = await embed_chunks(chunks)

        doc_id = uuid.uuid4().hex
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_name, "doc_id": doc_id} for _ in chunks]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        await bot.send_message(
            message.chat.id,
            f"Документ «{file_name}» сохранён в память ({len(chunks)} фрагментов). "
            f"Теперь задавай вопрос по нему.",
        )

    except Exception as exc:  # noqa: BLE001 — обработчик не должен ронять бота
        logger.exception("Document processing failed")
        await bot.send_message(message.chat.id, f"Ошибка обработки документа: {exc}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.message(F.text)
async def handle_question(message: Message, bot: Bot) -> None:
    """Ответить на вопрос пользователя на основе найденного контекста."""
    question = (message.text or "").strip()
    if not question or question.startswith("/"):
        return

    if collection.count() == 0:
        await bot.send_message(
            message.chat.id,
            "Память пуста. Сначала пришли мне документ (PDF, TXT или DOCX).",
        )
        return

    # Индикатор "печатает..." (aiogram 3.x не имеет метода delete_message для
    # chat_action, поэтому просто отправляем действие)
    try:
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    except (TelegramRetryAfter, TelegramAPIError):
        pass

    try:
        contexts = await retrieve_context(question)

        if not contexts:
            await bot.send_message(
                message.chat.id,
                "Не удалось найти релевантные фрагменты. Задай другой вопрос.",
            )
            return

        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
        answer = await answer_question(question, contexts)
        await _safe_send(bot, message.chat.id, answer)

    except RuntimeError as exc:
        logger.error("RAG pipeline error for user %s: %s", message.chat.id, exc)
        await bot.send_message(message.chat.id, f"Ошибка: {exc}")


# ─── Точка входа ─────────────────────────────────────────────────────────────


async def main() -> None:
    """Проверить ENV-переменные и запустить поллинг бота."""
    bot_token = os.environ.get("TG_BOT_API_KEY", "")
    if not bot_token:
        raise RuntimeError("TG_BOT_API_KEY not set")

    if not os.environ.get("PROXYAPI_API_KEY", ""):
        raise RuntimeError("PROXYAPI_API_KEY not set")

    bot = Bot(token=bot_token)
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Бот с долгой памятью запущен. Папка ChromaDB: %s", os.path.abspath(MEMORY_DIR))

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")
