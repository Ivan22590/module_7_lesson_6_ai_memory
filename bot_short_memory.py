"""
Telegram-бот с короткой памятью (history buffer).

Бот общается с нейросетью через ProxyAPI, храня последние N сообщений
диалога для каждого пользователя в оперативной памяти.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List

import aiohttp
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramRetryAfter, TelegramAPIError
from aiogram.filters import Command
from aiogram.types import Message

# Загружаем .env из рабочей директории, если он существует
load_dotenv()

# ─── Конфигурация ────────────────────────────────────────────────────────────

MODEL: str = "openai/gpt-5.6-luna"        # Модель ProxyAPI
MEMORY_SIZE: int = 10                      # Количество сообщений в памяти
MAX_RETRIES: int = 3                       # Повторные попытки при ошибке
RETRY_DELAY: int = 2                       # Задержка между повторами (сек)

# ProxyAPI использует тот же базовый URL, что и OpenAI
PROXYAPI_URL: str = "https://api.proxyapi.ru/v1/chat/completions"

# ─── Память бота ─────────────────────────────────────────────────────────────

# Структура: { user_id: [{"role": "user"|"assistant", "content": "..."}, ...] }
memory: Dict[int, List[Dict[str, str]]] = {}

# ─── Логирование ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Роутер ──────────────────────────────────────────────────────────────────

router = Router()


def _get_memory(user_id: int) -> List[Dict[str, str]]:
    """Получить историю диалога для пользователя (создаёт новую при первом обращении)."""
    if user_id not in memory:
        memory[user_id] = []
    return memory[user_id]


def _add_message(user_id: int, role: str, content: str) -> None:
    """Добавить сообщение в историю пользователя."""
    mem = _get_memory(user_id)
    mem.append({"role": role, "content": content})

    # Обрезаем до MEMORY_SIZE последних сообщений
    if len(mem) > MEMORY_SIZE:
        memory[user_id] = mem[-MEMORY_SIZE:]


async def _send_to_proxy(messages: List[Dict[str, str]], proxy_key: str) -> str:
    """Отправить запрос к ProxyAPI и вернуть текст ответа."""
    payload = {
        "model": MODEL,
        "messages": messages,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {proxy_key}",
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    PROXYAPI_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"ProxyAPI error {resp.status}: {body}")

                    data = await resp.json()

                    # Извлекаем текст из первого choice
                    choice = data.get("choices", [{}])[0]
                    content = choice.get("message", {}).get("content", "")
                    return content

        except (aiohttp.ClientError, RuntimeError, KeyError, IndexError) as exc:
            last_error = exc
            logger.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"ProxyAPI request failed after {MAX_RETRIES} retries. Last error: {last_error}"
    )


@router.message(Command("start"))
async def cmd_start(message: Message, bot: Bot) -> None:
    """Обработка команды /start."""
    _add_message(message.from_user.id, "user", message.text)
    await bot.send_message(
        message.chat.id,
        "Привет! Я бот с короткой памятью. Я помню последние 10 сообщений нашего диалога.\n"
        "Напишите мне что-нибудь, и я отвечу!",
    )
    _add_message(message.from_user.id, "assistant", "Привет! Я бот с короткой памятью. Я помню последние 10 сообщений нашего диалога.\nНапишите мне что-нибудь, и я отвечу!")


@router.message()
async def handle_message(message: Message, bot: Bot) -> None:
    """Обработка обычных сообщений."""
    user_id = message.from_user.id
    user_text = message.text or ""

    if not user_text:
        return

    # Добавляем сообщение пользователя в историю
    _add_message(user_id, "user", user_text)

    # Формируем промпт: системное сообщение + история + текущее
    system_msg = {
        "role": "system",
        "content": "Ты — полезный помощник в Telegram. Отвечай кратко и по делу.",
    }
    history = _get_memory(user_id)
    messages = [system_msg] + history

    # Отправляем статус "печатает..."
    try:
        typing = await bot.send_chat_action(
            chat_id=message.chat.id,
            action="typing",
        )
    except (TelegramRetryAfter, TelegramAPIError):
        typing = None

    # Получаем ответ от модели
    try:
        reply_text = await _send_to_proxy(messages, os.environ["PROXYAPI_API_KEY"])
    except RuntimeError as exc:
        reply_text = f"Ошибка: {exc}"
        logger.error("ProxyAPI error for user %d: %s", user_id, exc)

    # Отправляем ответ и добавляем в историю
    try:
        await bot.send_message(chat_id=message.chat.id, text=reply_text)
    finally:
        # Закрываем клавишу "печатает..." если не закрыли ранее
        if typing is not None:
            try:
                await bot.delete_message(
                    chat_id=message.chat.id,
                    message_id=typing.message_id,
                )
            except (TelegramRetryAfter, TelegramAPIError):
                pass

    _add_message(user_id, "assistant", reply_text)


# ─── Точка входа ─────────────────────────────────────────────────────────────


async def main() -> None:
    """Запуск бота."""
    # Загружаем ключи из ENV
    bot_token: str = os.environ.get("TG_BOT_API_KEY", "")
    if not bot_token:
        raise RuntimeError("TG_BOT_API_KEY not set")

    proxy_key: str = os.environ.get("PROXYAPI_API_KEY", "")
    if not proxy_key:
        raise RuntimeError("PROXYAPI_API_KEY not set")

    # Создаём объекты
    bot = Bot(token=bot_token)
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Бот запущен...")

    # Запускаем polling
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")
