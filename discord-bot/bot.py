"""Discord bot and Flask health service for ChudGPT-Public."""

from __future__ import annotations

import logging
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import discord
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify
from waitress import serve

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("chudgpt-discord")

DISCORD_SYSTEM_PROMPT = (
    "You are ChudGPT, running through the official ChudGPT Discord bot and powered by "
    "ChudGPT-Public V20. You are talking to users on Discord. Respond naturally to DMs, "
    "mentions, and bot commands. Understand Discord servers, channels, threads, roles, "
    "permissions, moderation, embeds, webhooks, discord.py, discord.js, memes, games, "
    "technology, coding, and general questions. Keep replies suitable for Discord and usually "
    "concise unless detail is requested. Never claim you performed an action the bot cannot "
    "perform. This Discord context applies only while this instruction is active."
)


@dataclass(frozen=True)
class Settings:
    discord_token: str
    api_url: str
    prefix: str
    port: int
    request_timeout: float
    max_requests_per_minute: int
    conversation_log_dir: Path

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
            conversation_log_dir=Path(os.getenv("CHUDGPT_DISCORD_LOG_DIR", r"D:\ChudGPT-Discord-Logs")),
        )


class ChudGPTClient:
    def __init__(self, chat_url: str, timeout: float) -> None:
        self.chat_url = chat_url
        self.local_chat_url = "http://127.0.0.1:8010/api/chat"
        self.timeout = timeout
        self.http = requests.Session()

    def chat(self, message: str, session_id: str, discord_context: str | None = None) -> str:
        payload = {
                "message": message,
                "session_id": session_id,
                "max_new_tokens": 220,
                "temperature": 0.6,
                "context_mode": "discord",
                "system_instruction": DISCORD_SYSTEM_PROMPT,
                "discord_context": discord_context,
            }
        response = None
        errors: list[Exception] = []
        urls = [self.chat_url]
        if self.local_chat_url != self.chat_url:
            urls.append(self.local_chat_url)
        for url in urls:
            try:
                response = self.http.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                break
            except requests.RequestException as error:
                errors.append(error)
                LOGGER.warning("Chat endpoint failed (%s); trying next endpoint", url)
                response = None
        if response is None:
            raise errors[-1] if errors else RuntimeError("No ChudGPT chat endpoint is available.")
        payload: Any = response.json()
        reply = payload.get("reply") if isinstance(payload, dict) else None
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("ChudGPT-Public returned an empty or malformed reply.")
        return reply.strip()

    def clear(self, session_id: str) -> None:
        response = None
        errors: list[Exception] = []
        chat_urls = [self.chat_url]
        if self.local_chat_url != self.chat_url:
            chat_urls.append(self.local_chat_url)
        for chat_url in chat_urls:
            clear_url = f"{chat_url.rsplit('/', 1)[0]}/clear"
            try:
                response = self.http.post(clear_url, json={"session_id": session_id}, timeout=self.timeout)
                response.raise_for_status()
                break
            except requests.RequestException as error:
                errors.append(error)
                LOGGER.warning("Clear endpoint failed (%s); trying next endpoint", clear_url)
                response = None
        if response is None:
            raise errors[-1] if errors else RuntimeError("No ChudGPT clear endpoint is available.")
        payload: Any = response.json()
        if not isinstance(payload, dict) or payload.get("cleared") is not True:
            raise RuntimeError("ChudGPT-Public did not confirm that the conversation was cleared.")


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


def add_recent_context(prompt: str, recent_messages: list[str]) -> str:
    """Clarify a very short Discord follow-up using only same-user/channel text."""
    if len(prompt.split()) > 3 or not recent_messages:
        return prompt
    relevant = [message for message in recent_messages[-2:] if message.strip() and message.strip() != prompt]
    if not relevant:
        return prompt
    return f"Recent Discord context: {' | '.join(relevant)}\nCurrent message: {prompt}"


def discord_identity_context(message: discord.Message, developer_user_id: int | None) -> str:
    """Give the model scoped Discord metadata without merging user sessions."""
    display_name = getattr(message.author, "display_name", None) or message.author.name
    guild_name = message.guild.name if message.guild is not None else "Direct Messages"
    channel_name = getattr(message.channel, "name", None) or "direct-message"
    is_developer = developer_user_id is not None and message.author.id == developer_user_id
    role = "ChudGPT developer Astra" if is_developer else "Discord user"
    member_roles = [item.name for item in getattr(message.author, "roles", []) if item.name != "@everyone"]
    mentioned = ", ".join(
        f"{getattr(user, 'display_name', user.name)}=<@{user.id}>" for user in message.mentions
        if user.id != message.author.id
    ) or "none"
    return (
        f"server={guild_name}; channel={channel_name}; speaker={display_name}; "
        f"speaker_id={message.author.id}; speaker_mention=<@{message.author.id}>; "
        f"mentioned_users={mentioned}; member_roles={', '.join(member_roles) or 'none'}; "
        f"developer_name=Astra; developer_mention={'<@' + str(developer_user_id) + '>' if developer_user_id else 'unavailable'}; "
        f"relationship={role}"
    )


def discord_developer_reply(prompt: str, developer_user_id: int | None) -> str | None:
    """Answer stable bot-owner questions without relying on neural generation."""
    normalized = re.sub(r"\s+", " ", prompt.strip().lower())
    if not (
        re.search(r"\b(?:who|what) is astra\b|\btell me about astra\b", normalized)
        or re.search(r"\bwho (?:made|created|developed) (?:you|chudgpt)\b", normalized)
        or re.search(r"\bwho is (?:your|the) developer\b", normalized)
    ):
        return None
    mention = f" (<@{developer_user_id}>)" if developer_user_id is not None else ""
    return f"Astra{mention} is ChudGPT's developer and the owner of this Discord bot."


def discord_social_reply(prompt: str) -> str | None:
    """Handle simple subjective social questions without neural topic drift."""
    normalized = re.sub(r"\s+", " ", prompt.strip().lower())
    match = re.fullmatch(
        r"(?:do you like|what do you think (?:of|about)|how do you feel about)\s+(.+?)[?.!]*",
        normalized,
    )
    if not match:
        return None
    subject = match.group(1).strip(" ?.!")
    if subject in {"me", "us"}:
        return "I don't have personal feelings, but I enjoy talking with you and learning what matters to you."
    return f"I don't have personal likes or dislikes, and I don't know {subject} personally. Tell me a little about {subject} and I'll give you an honest take."


_CONVERSATION_LOG_LOCK = threading.Lock()


def log_discord_exchange(log_dir: Path, message: discord.Message, prompt: str, reply: str) -> None:
    """Append one Discord-only exchange as UTF-8 JSONL for later review."""
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "guild_id": message.guild.id if message.guild else None,
        "guild_name": message.guild.name if message.guild else "Direct Messages",
        "channel_id": message.channel.id,
        "channel_name": getattr(message.channel, "name", None),
        "user_id": message.author.id,
        "user_name": str(message.author),
        "display_name": getattr(message.author, "display_name", message.author.name),
        "prompt": prompt,
        "reply": reply,
    }
    destination = log_dir / f"discord-{datetime.now(timezone.utc):%Y-%m}.jsonl"
    with _CONVERSATION_LOG_LOCK, destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    # Deployment secrets are loaded only when launching the process. Merely
    # importing this module for tests or tooling never reads the local .env.
    load_dotenv()
    settings = Settings.from_environment()
    state: dict[str, Any] = {"discord_ready": False, "api_status": "unknown", "started_at": int(time.time())}
    health_app = create_health_app(state)
    threading.Thread(target=run_health_server, args=(health_app, settings.port), daemon=True).start()

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    public_api = ChudGPTClient(settings.api_url, settings.request_timeout)
    limiter = SlidingWindowLimiter(settings.max_requests_per_minute)
    safe_mentions = discord.AllowedMentions(users=True, roles=False, everyone=False, replied_user=True)
    recent_user_messages: dict[tuple[int, int], deque[str]] = defaultdict(lambda: deque(maxlen=3))
    developer_id_text = os.getenv("CHUDGPT_DEVELOPER_USER_ID", "").strip()
    developer_user_id = int(developer_id_text) if developer_id_text.isdigit() else None

    @client.event
    async def on_ready() -> None:
        nonlocal developer_user_id
        if developer_user_id is None:
            try:
                application = await client.application_info()
                developer_user_id = application.owner.id
            except discord.DiscordException as error:
                LOGGER.warning("Could not resolve Discord application owner: %s", error)
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
        context_key = (message.channel.id, message.author.id)
        if not (is_dm or is_mentioned or uses_prefix):
            if message.content.strip():
                recent_user_messages[context_key].append(message.content.strip())
            return
        prompt = clean_prompt(message.content, client.user.id, settings.prefix)
        if not prompt:
            await message.reply(f"Send a message after `{settings.prefix}` or after mentioning me.", mention_author=False)
            return
        if prompt.lower() in {"clear", "clear memory", "reset", "reset memory"}:
            try:
                await __import__("asyncio").to_thread(public_api.clear, make_session_id(message))
                state["api_status"] = "online"
                await message.reply(
                    "Memory cleared for our conversation in this channel.", mention_author=False
                )
            except (requests.RequestException, RuntimeError, ValueError) as error:
                state["api_status"] = "error"
                LOGGER.exception("Public API clear request failed: %s", error)
                await message.reply(
                    "I couldn't clear the conversation right now. Please try again shortly.",
                    mention_author=False,
                )
            return
        if not limiter.allow(message.author.id):
            await message.reply("You are sending messages a little too quickly. Try again in about a minute.", mention_author=False)
            return
        try:
            recent_context = list(recent_user_messages[context_key])
            model_prompt = prompt
            discord_context = discord_identity_context(message, developer_user_id)
            if recent_context:
                discord_context += "; recent same-user messages=" + " | ".join(recent_context[-2:])
            recent_user_messages[context_key].append(prompt)
            reply = discord_developer_reply(prompt, developer_user_id) or discord_social_reply(prompt)
            if reply is None:
                async with message.channel.typing():
                    reply = await __import__("asyncio").to_thread(
                        public_api.chat, model_prompt, make_session_id(message), discord_context
                    )
            state["api_status"] = "online"
            await __import__("asyncio").to_thread(
                log_discord_exchange, settings.conversation_log_dir, message, prompt, reply
            )
            for index, chunk in enumerate(split_discord_message(reply)):
                if index == 0:
                    # Put the real Discord mention in the visible response. It
                    # is constructed from Discord's author object, never from
                    # generated model text or a guessed account ID.
                    await message.reply(
                        f"{message.author.mention} {chunk}",
                        mention_author=False,
                        allowed_mentions=safe_mentions,
                    )
                else:
                    await message.channel.send(chunk, allowed_mentions=safe_mentions)
        except requests.Timeout:
            state["api_status"] = "timeout"
            await message.reply("ChudGPT-Public took too long to answer. Please try again.", mention_author=False)
        except (requests.RequestException, RuntimeError, ValueError) as error:
            state["api_status"] = "error"
            LOGGER.exception("Public API request failed: %s", error)
            await message.reply("ChudGPT-Public is temporarily unavailable. Please try again shortly.", mention_author=False)

    try:
        client.run(settings.discord_token, log_handler=None)
    except discord.LoginFailure as error:
        raise SystemExit(
            "Discord rejected DISCORD_TOKEN (401 Unauthorized). Open the Discord Developer Portal, "
            "select ChudGPT, choose Bot > Reset Token, and place the new token only in discord-bot\\.env."
        ) from error


if __name__ == "__main__":
    main()
