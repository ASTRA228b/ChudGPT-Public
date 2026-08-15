"""Discord bot and Flask health service for ChudGPT-Public."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import discord
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify
from waitress import serve

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("chudgpt-discord")


@dataclass(frozen=True)
class Settings:
    discord_token: str
    api_url: str
    prefix: str
    port: int
    request_timeout: float
    max_requests_per_minute: int

    @classmethod
    def from_environment(cls) -> "Settings":
        token = os.getenv("DISCORD_TOKEN", "").strip()
        if not token:
            raise RuntimeError("DISCORD_TOKEN is missing. Copy .env.example to .env and add the bot token.")
        api_root = os.getenv("CHUDGPT_PUBLIC_API_URL", "https://chudgpt-public.vercel.app/api").rstrip("/")
        return cls(
            discord_token=token,
            api_url=f"{api_root}/chat",
            prefix=os.getenv("BOT_PREFIX", "!chud").strip() or "!chud",
            port=int(os.getenv("PORT", "8080")),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "90")),
            max_requests_per_minute=max(1, int(os.getenv("MAX_REQUESTS_PER_MINUTE", "8"))),
        )


class ChudGPTClient:
    def __init__(self, chat_url: str, timeout: float) -> None:
        self.chat_url = chat_url
        self.timeout = timeout
        self.http = requests.Session()

    def chat(self, message: str, session_id: str) -> str:
        response = self.http.post(
            self.chat_url,
            json={"message": message, "session_id": session_id, "max_new_tokens": 220, "temperature": 0.6},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: Any = response.json()
        reply = payload.get("reply") if isinstance(payload, dict) else None
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("ChudGPT-Public returned an empty or malformed reply.")
        return reply.strip()


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: dict[int, deque[float]] = defaultdict(deque)

    def allow(self, user_id: int) -> bool:
        now = time.monotonic()
        timestamps = self.requests[user_id]
        while timestamps and timestamps[0] <= now - self.window_seconds:
            timestamps.popleft()
        if len(timestamps) >= self.limit:
            return False
        timestamps.append(now)
        return True


def make_session_id(message: discord.Message) -> str:
    """Keep users isolated while retaining context inside each channel."""
    location = message.guild.id if message.guild else "dm"
    return f"discord-{location}-{message.channel.id}-{message.author.id}"


def clean_prompt(content: str, bot_user_id: int | None, prefix: str) -> str:
    cleaned = content.strip()
    if bot_user_id is not None:
        cleaned = re.sub(rf"<@!?{bot_user_id}>", "", cleaned).strip()
    if cleaned.lower().startswith(prefix.lower()):
        cleaned = cleaned[len(prefix):].strip()
    return cleaned


def split_discord_message(text: str, limit: int = 1_900) -> list[str]:
    """Split safely below Discord's message limit, preferring line breaks."""
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def create_health_app(state: dict[str, Any]) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> tuple[dict[str, Any], int]:
        return {
            "name": "ChudGPT-Public Discord Bot",
            "status": "online" if state.get("discord_ready") else "starting",
            "usage": "DM the bot, mention it, or use the configured !chud prefix.",
        }, 200

    @app.get("/health")
    def health() -> tuple[dict[str, Any], int]:
        ready = bool(state.get("discord_ready"))
        return {"ok": ready, "discord_ready": ready, "api": state.get("api_status", "unknown")}, 200 if ready else 503

    @app.get("/status")
    def status() -> tuple[Any, int]:
        return jsonify({key: value for key, value in state.items() if key != "token"}), 200

    return app


def run_health_server(app: Flask, port: int) -> None:
    serve(app, host="0.0.0.0", port=port, threads=4)


def main() -> None:
    settings = Settings.from_environment()
    state: dict[str, Any] = {"discord_ready": False, "api_status": "unknown", "started_at": int(time.time())}
    health_app = create_health_app(state)
    threading.Thread(target=run_health_server, args=(health_app, settings.port), daemon=True).start()

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    public_api = ChudGPTClient(settings.api_url, settings.request_timeout)
    limiter = SlidingWindowLimiter(settings.max_requests_per_minute)

    @client.event
    async def on_ready() -> None:
        state["discord_ready"] = True
        state["bot_user"] = str(client.user)
        state["guild_count"] = len(client.guilds)
        LOGGER.info("Logged in as %s in %d guild(s)", client.user, len(client.guilds))
        await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name=f"{settings.prefix} or mentions"))

    @client.event
    async def on_disconnect() -> None:
        state["discord_ready"] = False

    @client.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot or client.user is None:
            return
        is_dm = message.guild is None
        is_mentioned = client.user in message.mentions
        uses_prefix = message.content.lower().startswith(settings.prefix.lower())
        if not (is_dm or is_mentioned or uses_prefix):
            return
        prompt = clean_prompt(message.content, client.user.id, settings.prefix)
        if not prompt:
            await message.reply(f"Send a message after `{settings.prefix}` or after mentioning me.", mention_author=False)
            return
        if not limiter.allow(message.author.id):
            await message.reply("You are sending messages a little too quickly. Try again in about a minute.", mention_author=False)
            return
        try:
            async with message.channel.typing():
                reply = await __import__("asyncio").to_thread(public_api.chat, prompt, make_session_id(message))
            state["api_status"] = "online"
            for index, chunk in enumerate(split_discord_message(reply)):
                if index == 0:
                    await message.reply(chunk, mention_author=False)
                else:
                    await message.channel.send(chunk)
        except requests.Timeout:
            state["api_status"] = "timeout"
            await message.reply("ChudGPT-Public took too long to answer. Please try again.", mention_author=False)
        except (requests.RequestException, RuntimeError, ValueError) as error:
            state["api_status"] = "error"
            LOGGER.exception("Public API request failed: %s", error)
            await message.reply("ChudGPT-Public is temporarily unavailable. Please try again shortly.", mention_author=False)

    client.run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
