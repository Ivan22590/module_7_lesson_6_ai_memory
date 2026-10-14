"""
Telegram-бот с гибридной памятью: короткая (history buffer) + долгая (RAG).

Объединяет два подхода:
1. Короткая память — последние N сообщений диалога каждого пользователя
   хранятся в оперативной памяти (dict).
2. Долгая память — документы пользователя (PDF / TXT / DOCX) разбиваются на
   чанки, превращаются в эмбеддинги и сохраняются в локальную ChromaDB.

Перед ответом модель получает: системную инструкцию, релевантные фрагменты
документов (долгая память) и историю диалога (короткая память).
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

MEMORY_DIR: str = "./memory"      # Persistent storage ChromaDB
MEMORY_SIZE: int = 10             # Сколько сообщений короткой памяти хранить
CHUNK_SIZE: int = 500             # Размер чанка в символах
CHUNK_OVERLAP: int = 50           # Перекрытие соседних чанков
TOP_K: int = 3                    # Сколько фрагментов документа подтягивать
EMBED_BATCH: int = 64             # Чанков за один запрос к /embeddings
MAX_RETRIES: int = 3              # Повторные попытки при ошибке API
RETRY_DELAY: int = 2              # Задержка между повторами (сек)

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

# ─── Короткая память (в оперативке) ──────────────────────────────────────────

# Структура: { user_id: [{"role": "user"|"assistant", "content": "..."}, ...] }
history: Dict[int, List[Dict[str, str]]] = {}

# ─── Роутер ──────────────────────────────────────────────────────────────────

router = Router()


# ─── Короткая память ─────────────────────────────────────────────────────────


def get_history(user_id: int) -> List[Dict[str, str]]:
    """Вернуть историю диалога пользователя (пустой список при первом обращении)."""
    return history.get(user_id, [])


def add_message(user_id: int, role: str, content: str) -> None:
    """Добавить сообщение в короткую память и обрезать до MEMORY_SIZE последних.

    Args:
        user_id: Идентификатор пользователя Telegram.
        role: Роль сообщения — "user" или "assistant".
        content: Текст сообщения.
    """
    messages = history.setdefault(user_id, [])
    messages.append({"role": role, "content": content})

    if len(messages) > MEMORY_SIZE:
        history[user_id] = messages[-MEMORY_SIZE:]


# ─── Долгая память: 1. чтение документа ──────────────────────────────────────


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
        return "\n".join(para.text for para in doc.paragraphs)

    raise ValueError(f"Неподдерживаемый формат: {ext}")


# ─── 2. Разбиение на чанки ───────────────────────────────────────────────────


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Разбить текст на перекрывающиеся фрагменты размером ~chunk_size символов.

    Границы чанков стараемся ставить на конце предложения, чтобы не терять смысл.

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

        start = max(end - overlap, start + 1)

    return [chunk for chunk in chunks if chunk]


# ─── 3. Эмбеддинги ───────────────────────────────────────────────────────────


async def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Получить векторы эмбеддингов для списка чанков через ProxyAPI.

    Запросы отправляются батчами по EMBED_BATCH чанков.

    Args:
        chunks: Список текстовых фрагментов.

    Returns:
        Список эмбеддингов (по одному вектору на каждый чанк).

    Raises:
        RuntimeError: Если API недоступен после MAX_RETRIES попыток.
    """
    api_key = os.environ["PROXYAPI_API_KEY"]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    all_embeddings: List[List[float]] = []

    for offset in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[offset : offset + EMBED_BATCH]
        payload: Dict[str, Any] = {"model": EMBEDDING_MODEL, "input": batch}

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
                all_embeddings.extend(item["embedding"] for item in items)
                break

            except (aiohttp.ClientError, RuntimeError, KeyError) as exc:
                last_error = exc
                logger.warning("Embedding attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
        else:
            raise RuntimeError(
                f"Не удалось получить эмбеддинги после {MAX_RETRIES} попыток: {last_error}"
            )

    return all_embeddings


# ─── 4. Поиск релевантного контекста (долгая память) ─────────────────────────


async def retrieve_context(user_id: int, question: str, top_k: int = TOP_K) -> List[str]:
    """Найти релевантные фрагменты документов конкретного пользователя.

    Эмбеддинг вопроса сравнивается с сохранёнными векторами по косинусному
    расстоянию, фильтр по user_id не даёт смешивать документы разных людей.

    Args:
        user_id: Идентификатор пользователя Telegram.
        question: Текст вопроса.
        top_k: Количество возвращаемых фрагментов.

    Returns:
        Список текстов релевантных чанков (от наиболее к наименее похожему).
        Пустой список, если у пользователя нет документов.
    """
    user_filter = {"user_id": str(user_id)}

    # Быстрая проверка: есть ли у пользователя документы
    existing = collection.get(where=user_filter, limit=1)
    if not existing["ids"]:
        return []

    query_embeddings = await embed_chunks([question])

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
        where=user_filter,
    )

    documents: List[List[str]] = results.get("documents") or [[]]
    return [doc for doc in documents[0] if doc]


# ─── 5. Генерация ответа (короткая + долгая память) ─────────────────────────


async def answer_question(
    question: str,
    contexts: List[str],
    short_memory: List[Dict[str, str]],
) -> str:
    """Сгенерировать ответ модели с учётом документов и истории диалога.

    Формируется запрос вида: system (правила + фрагменты документов) +
    последние сообщения диалога (короткая память) + текущий вопрос.

    Args:
        question: Текущий вопрос пользователя.
        contexts: Релевантные фрагменты документов (долгая память).
        short_memory: История последних сообщений диалога.

    Returns:
        Текст ответа модели.

    Raises:
        RuntimeError: Если API недоступен после MAX_RETRIES попыток.
    """
    system_prompt = (
        "Ты — полезный ассистент в Telegram. У тебя два источника знаний:\n"
        "1. Фрагменты документов пользователя (долгая память) — основной источник "
        "фактов, не выдумывай их.\n"
        "2. История недавнего диалога (короткая память) — учитывай её, чтобы понимать "
        "контекст реплик («это», «он», «продолжи» и тому подобное).\n\n"
        "Если ответа нет ни в документах, ни в диалоге — так и скажи. "
        "Отвечай на языке пользователя, кратко и по делу."
    )

    if contexts:
        context_block = "\n\n".join(
            f"[Фрагмент {i + 1}]\n{chunk}" for i, chunk in enumerate(contexts)
        )
        system_prompt += f"\n\nФрагменты документов:\n{context_block}"
    else:
        system_prompt += "\n\nДокументы пользователем ещё не загружены."

    # История диалога без текущего сообщения (оно передаётся отдельно в question)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for item in short_memory:
        if item["role"] in ("user", "assistant"):
            messages.append({"role": item["role"], "content": item["content"]})

    messages.append({"role": "user", "content": question})

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


async def safe_send(bot: Bot, chat_id: int, text: str) -> None:
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
        "Привет! Я бот с гибридной памятью: короткой (история диалога) "
        "и долгой (документы).\n\n"
        f"• Обычные сообщения — я помню последние {MEMORY_SIZE} реплик нашего диалога.\n"
        "• Пришли документ (PDF, TXT или DOCX) — я сохраню его в векторную базу ChromaDB "
        "и буду отвечать по нему.\n"
        "• На любой вопрос я объединю историю диалога + фрагменты твоих документов.\n\n"
        "Команды:\n"
        "/forget — забыть историю диалога (короткая память)\n"
        "/clear — удалить мои документы (долгая память)\n"
        "/memory — показать состояние памяти",
    )


@router.message(Command("forget"))
async def cmd_forget(message: Message, bot: Bot) -> None:
    """Очистить короткую память (историю диалога) пользователя."""
    history.pop(message.from_user.id, None)
    await bot.send_message(
        message.chat.id, "Короткая память очищена — история диалога забыта."
    )


@router.message(Command("clear"))
async def cmd_clear(message: Message, bot: Bot) -> None:
    """Удалить все документы пользователя из ChromaDB (долгая память)."""
    user_docs = collection.get(where={"user_id": str(message.from_user.id)})

    if user_docs["ids"]:
        collection.delete(ids=user_docs["ids"])
        await bot.send_message(
            message.chat.id,
            f"Долгая память очищена: удалено фрагментов — {len(user_docs['ids'])}.",
        )
    else:
        await bot.send_message(message.chat.id, "У тебя в базе нет документов.")


@router.message(Command("memory"))
async def cmd_memory(message: Message, bot: Bot) -> None:
    """Показать текущее состояние короткой и долгой памяти пользователя."""
    user_id = message.from_user.id
    user_docs = collection.get(where={"user_id": str(user_id)})

    await bot.send_message(
        message.chat.id,
        f"Короткая память: {len(get_history(user_id))} сообщений из {MEMORY_SIZE}.\n"
        f"Долгая память: {len(user_docs['ids'])} фрагментов документов в ChromaDB.",
    )


@router.message(F.document)
async def handle_document(message: Message, bot: Bot) -> None:
    """Принять документ: скачать, извлечь текст, построить эмбеддинги, сохранить."""
    user_id = message.from_user.id
    document = message.document
    file_name = document.file_name or "document.bin"
    ext = os.path.splitext(file_name)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        await bot.send_message(
            message.chat.id,
            f"Неподдерживаемый формат «{ext or 'без расширения'}». "
            f"Пришли файл: {', '.join(SUPPORTED_EXTENSIONS)}.",
        )
        return

    temp_path = f"temp_{uuid.uuid4().hex}{ext}"

    try:
        await bot.send_message(message.chat.id, f"Загружаю «{file_name}»...")

        # Скачиваем файл из Telegram во временный путь
        file_info = await bot.get_file(document.file_id)
        await bot.download_file(file_info.file_path, temp_path)

        text = load_document(temp_path)
        if not text.strip():
            await bot.send_message(
                message.chat.id,
                "В документе не удалось найти текст "
                "(возможно, это скан без распознавания).",
            )
            return

        chunks = chunk_text(text)
        await bot.send_message(
            message.chat.id,
            f"Извлечено {len(text)} символов, фрагментов: {len(chunks)}. "
            f"Строю эмбеддинги...",
        )

        embeddings = await embed_chunks(chunks)

        # Сохраняем с метаданными, включая user_id — так документы не смешиваются
        doc_id = uuid.uuid4().hex
        collection.upsert(
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"source": file_name, "doc_id": doc_id, "user_id": str(user_id)}
            ]
            * len(chunks),
        )

        await bot.send_message(
            message.chat.id,
            f"Документ «{file_name}» сохранён в долгую память ({len(chunks)} фрагментов). "
            f"Теперь задавай вопросы — я учту и его, и историю нашего диалога.",
        )

    except Exception as exc:  # noqa: BLE001 — обработчик не должен ронять бота
        logger.exception("Document processing failed")
        await bot.send_message(message.chat.id, f"Ошибка обработки документа: {exc}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.message(F.text)
async def handle_message(message: Message, bot: Bot) -> None:
    """Основной обработчик: короткая память + долгая память → ответ модели."""
    user_id = message.from_user.id
    text = (message.text or "").strip()

    if not text or text.startswith("/"):
        return

    # 1. Кладём реплику пользователя в короткую память
    add_message(user_id, "user", text)

    try:
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    except (TelegramRetryAfter, TelegramAPIError):
        pass

    # 2. Ищем релевантные фрагменты документов (долгая память)
    # 3. Отдаём модели историю диалога + контекст + текущий вопрос
    try:
        contexts = await retrieve_context(user_id, text)
        reply = await answer_question(
            question=text,
            contexts=contexts,
            short_memory=get_history(user_id)[:-1],  # без текущего сообщения
        )
    except RuntimeError as exc:
        reply = f"Ошибка: {exc}"
        logger.error("Memory pipeline error for user %d: %s", user_id, exc)

    # 4. Отправляем ответ и кладём его в короткую память
    await safe_send(bot, message.chat.id, reply)
    add_message(user_id, "assistant", reply)


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

    logger.info(
        "Бот с гибридной памятью запущен. ChromaDB: %s", os.path.abspath(MEMORY_DIR)
    )

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")
