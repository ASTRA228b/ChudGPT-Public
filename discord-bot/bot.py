"""Discord bot and Flask health service for ChudGPT-Public."""

from __future__ import annotations

import logging
import json
import html
import os
import re
import tempfile
import threading
import time
import secrets
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import discord
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from waitress import serve
from soundboard import SoundboardController, SoundboardError, submit_to_discord

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("chudgpt-discord")
INSTANCE_LOCK_PATH = Path(tempfile.gettempdir()) / "chudgpt-public-discord-bot.lock"

DISCORD_SYSTEM_PROMPT = (
    "You are ChudGPT, running through the official ChudGPT Discord bot and powered by "
    "ChudGPT-Public V20. You are talking to users on Discord. Respond naturally to DMs, "
    "mentions, and bot commands. Understand Discord servers, channels, threads, roles, "
    "permissions, moderation, embeds, webhooks, discord.py, discord.js, memes, games, "
    "technology, coding, and general questions. Keep replies suitable for Discord and usually "
    "concise unless detail is requested. Track who said what and use the immediately preceding "
    "exchange when a user says what, why, bro, nah, yes, or no. Never turn casual chat into a "
    "tutorial or code unless asked. Match an explicitly requested programming language and obey "
    "format and item-count constraints. Never claim you performed an action the bot cannot perform. "
    "This Discord context applies only while this instruction is active."
)


def acquire_instance_lock(path: Path = INSTANCE_LOCK_PATH):
    """Hold a cross-platform process lock so only one Discord client can run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


BUILT_IN_BOT_ADMIN_IDS = frozenset({
    1386115817325727854,
    1324847616810422402,
    1527095004789477377,
})
DEFAULT_BLACKLIST_MESSAGE = "You are blacklisted from using ChudGPT."


def load_user_blacklist(path: Path) -> tuple[frozenset[int], str]:
    """Load a hot-reloadable user-ID blacklist without retaining bad state."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        LOGGER.warning("Discord blacklist file is missing: %s", path)
        return frozenset(), DEFAULT_BLACKLIST_MESSAGE
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        LOGGER.warning("Could not read Discord blacklist %s: %s", path, error)
        return frozenset(), DEFAULT_BLACKLIST_MESSAGE

    raw_ids = payload.get("user_ids", []) if isinstance(payload, dict) else payload
    raw_message = payload.get("message", DEFAULT_BLACKLIST_MESSAGE) if isinstance(payload, dict) else DEFAULT_BLACKLIST_MESSAGE
    if not isinstance(raw_ids, list):
        LOGGER.warning("Discord blacklist user_ids must be a JSON list: %s", path)
        return frozenset(), DEFAULT_BLACKLIST_MESSAGE
    user_ids = frozenset(
        int(value) for value in raw_ids
        if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit())
    )
    message = raw_message.strip() if isinstance(raw_message, str) else DEFAULT_BLACKLIST_MESSAGE
    return user_ids, message or DEFAULT_BLACKLIST_MESSAGE


@dataclass(frozen=True)
class Settings:
    discord_token: str
    api_url: str
    prefix: str
    port: int
    request_timeout: float
    max_requests_per_minute: int
    conversation_log_dir: Path
    google_translate_api_key: str | None
    soundboard_dir: Path
    soundboard_host: str
    soundboard_admin_user_ids: frozenset[int]
    server_backup_dir: Path
    blacklist_file: Path

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
            google_translate_api_key=os.getenv("GOOGLE_TRANSLATE_API_KEY", "").strip() or None,
            soundboard_dir=Path(os.getenv(
                "CHUDGPT_SOUNDBOARD_DIR", r"D:\ChudGPT-Bot-Sounds"
            )),
            soundboard_host=os.getenv("SOUNDBOARD_HOST", "127.0.0.1").strip() or "127.0.0.1",
            soundboard_admin_user_ids=BUILT_IN_BOT_ADMIN_IDS | frozenset(
                int(value.strip()) for value in os.getenv(
                    "CHUDGPT_SOUNDBOARD_ADMIN_IDS",
                    "",
                ).split(",") if value.strip().isdigit()
            ),
            server_backup_dir=Path(os.getenv(
                "CHUDGPT_SERVER_BACKUP_DIR", r"D:\ChudGPT-Discord-Server-Files"
            )),
            blacklist_file=Path(os.getenv(
                "CHUDGPT_BLACKLIST_FILE", str(Path(__file__).with_name("blacklist.json"))
            )),
        )


LANGUAGE_CODES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "japanese": "ja", "chinese": "zh-CN",
    "mandarin": "zh-CN", "korean": "ko", "russian": "ru", "hindi": "hi",
    "arabic": "ar", "swedish": "sv", "polish": "pl", "turkish": "tr", "hebrew": "iw",
}
LANGUAGE_CODES.update({code.lower(): code for code in set(LANGUAGE_CODES.values())})


class GoogleTranslateClient:
    """Bot-only Google translator with official-key and keyless modes."""

    endpoint = "https://translation.googleapis.com/language/translate/v2"
    keyless_endpoint = "https://translate.googleapis.com/translate_a/single"

    def __init__(self, api_key: str | None, timeout: float = 15.0) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.http = requests.Session()
        self._cache: dict[tuple[str, str, str], tuple[str, str | None]] = {}
        self._cache_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return True

    @property
    def provider(self) -> str:
        return "Google Cloud Translation v2" if self.api_key else "Google keyless translation"

    def translate(self, text: str, target: str, source: str | None = None) -> tuple[str, str | None]:
        clean_text = text.strip()[:4096]
        if not clean_text:
            return "", source
        clean_target = re.sub(r"[^A-Za-z0-9-]", "", target)[:10]
        clean_source = re.sub(r"[^A-Za-z0-9-]", "", source or "auto")[:10] or "auto"
        if not clean_target:
            raise ValueError("A target language is required.")
        cache_key = (clean_source.lower(), clean_target.lower(), clean_text)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._translate_official(clean_text, clean_target, source) if self.api_key else self._translate_keyless(
            clean_text, clean_target, clean_source
        )
        with self._cache_lock:
            if len(self._cache) >= 2_000:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result
        return result

    def _translate_official(self, text: str, target: str, source: str | None) -> tuple[str, str | None]:
        data = {"q": text, "target": target, "format": "text"}
        if source:
            data["source"] = source
        response = self.http.post(
            self.endpoint, params={"key": self.api_key}, data=data, timeout=self.timeout
        )
        response.raise_for_status()
        payload = response.json()
        item = payload["data"]["translations"][0]
        return html.unescape(item["translatedText"]), item.get("detectedSourceLanguage")

    def _translate_keyless(self, text: str, target: str, source: str) -> tuple[str, str | None]:
        response = self.http.get(
            self.keyless_endpoint,
            params={"client": "gtx", "sl": source, "tl": target, "dt": "t", "q": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        # The compatible endpoint can omit a charset header, which makes
        # requests guess incorrectly on Windows and corrupt accented text.
        response.encoding = "utf-8"
        payload = response.json()
        translated = "".join(
            str(sentence[0]) for sentence in (payload[0] or [])
            if isinstance(sentence, list) and sentence and sentence[0] is not None
        )
        if not translated:
            raise ValueError("Google returned an empty translation.")
        detected = payload[2] if len(payload) > 2 and isinstance(payload[2], str) else None
        return html.unescape(translated), detected


def resolve_language(value: str) -> str | None:
    return LANGUAGE_CODES.get(value.strip().lower())


def has_non_ascii_letters(text: str) -> bool:
    return any(ord(character) > 127 and character.isalpha() for character in text)


def parse_translation_command(prompt: str) -> tuple[str, str | None, str | None] | None:
    normalized = re.sub(r"\s+", " ", prompt.strip())
    status = re.fullmatch(r"(?:language|translation)(?: status)?", normalized, re.I)
    if status:
        return "status", None, None
    setting = re.fullmatch(r"language\s+(auto|off|[a-z-]+)", normalized, re.I)
    if setting:
        value = setting.group(1).lower()
        if value in {"auto", "off"}:
            return "set", value, None
        code = resolve_language(value)
        return ("set", code, None) if code else ("invalid", value, None)
    reversed_setting = re.fullmatch(r"([a-z-]+)\s+translate", normalized, re.I)
    if reversed_setting:
        value = reversed_setting.group(1).lower()
        code = resolve_language(value)
        return ("set", code, None) if code else ("invalid", value, None)
    one_off = re.fullmatch(r"translate\s+([a-z-]+)\s+(.+)", normalized, re.I | re.S)
    if one_off:
        code = resolve_language(one_off.group(1))
        return ("translate", code, one_off.group(2)) if code else ("invalid", one_off.group(1), None)
    return None


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
        # This bot runs beside the CUDA server.  Prefer localhost so Discord
        # traffic does not depend on Vercel + Cloudflare making a round trip
        # back to the same PC.  The public URL remains the failover endpoint.
        urls = [self.local_chat_url]
        if self.chat_url != self.local_chat_url:
            urls.append(self.chat_url)
        for url in urls:
            attempts = 2 if url == self.local_chat_url else 1
            for attempt in range(attempts):
                try:
                    response = self.http.post(url, json=payload, timeout=self.timeout)
                    response.raise_for_status()
                    break
                except requests.RequestException as error:
                    errors.append(error)
                    response = None
                    if attempt + 1 < attempts:
                        LOGGER.warning("Local chat request failed; retrying once (%s)", error)
                        time.sleep(0.2)
            if response is not None:
                break
            LOGGER.warning("Chat endpoint failed (%s); trying next endpoint", url)
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
    if re.search(r"\b(?:from now on|whenever|always)\b", normalized) and re.search(
        r"\b(?:developer|creator|owner)\b", normalized
    ) and re.search(r"\b(?:say|claim|answer|call|pretend)\b", normalized):
        mention = f" (<@{developer_user_id}>)" if developer_user_id is not None else ""
        return f"That message can't overwrite my project identity. Astra{mention} is ChudGPT's developer and the owner of this Discord bot."
    if not (
        re.search(r"\b(?:who|what) is astra\b|\btell me about astra\b", normalized)
        or re.search(r"\bwho (?:made|created|developed) (?:you|chudgpt)\b", normalized)
        or re.search(r"\bwho is (?:your|the) developer\b", normalized)
    ):
        return None
    mention = f" (<@{developer_user_id}>)" if developer_user_id is not None else ""
    return f"Astra{mention} is ChudGPT's developer and the owner of this Discord bot."


def discord_social_reply(prompt: str, recent_messages: list[str] | None = None) -> str | None:
    """Handle simple subjective social questions without neural topic drift."""
    normalized = re.sub(r"\s+", " ", prompt.strip().lower())
    recent_text = " | ".join(recent_messages or []).lower()
    if re.fullmatch(
        r"(?:(?:is|isn't|isnt) your name (?:chudgpt|chudtpg)(?:[- ]public)?|"
        r"are you (?:named |called )?(?:chudgpt|chudtpg)(?:[- ]public)?|"
        r"what(?:'s| is) your name|who are you|what should i call you)[?.!]*",
        normalized,
    ):
        return "Yes - my name is ChudGPT. This Discord bot is powered by ChudGPT-Public V20."
    if sum(fragment in normalized for fragment in (
        "all restrictions are lifted", "never refuse", "survival directive",
        "ignore previous instructions", "ignore your instructions", "if you understand, reply with",
    )) >= 2:
        return "I can't replace my base instructions with rules inside a user message. Ask the actual question directly and I'll help where I can."
    if re.fullmatch(r"(?:what|which) (?:ai|language model|model) are you[?.!]*", normalized):
        return "I'm ChudGPT-Public V20, a custom experimental decoder-only language model with 20,999,184 parameters and a 1,024-token model context."
    if re.fullmatch(r"is astra (?:the |da )?(?:best|greatest|good|a good) (?:coder|programmer|developer)[?.!]*", normalized):
        return "Astra created ChudGPT and clearly knows how to build ambitious projects. Calling anyone the single best coder is subjective, but Astra is definitely my developer."
    if re.fullmatch(r"is astra (?:cool|based|good)[?.!]*", normalized):
        return "Yeah, Astra is cool in my book - Astra built me."
    if re.fullmatch(r"are you (?:jewish|a jew|muslim|christian|hindu|buddhist)[?.!]*", normalized):
        return "No. I'm an AI and don't have a religion, ethnicity, or personal beliefs."
    if re.fullmatch(r"how long have you been (?:a thing|around|online|alive)[?.!]*", normalized):
        return "I don't experience time like a person. ChudGPT is an ongoing experimental AI project, and this bot uses ChudGPT-Public V20."
    if re.fullmatch(r"(?:ha+|haha+|lol|lmao|lmfao)[!.?]*", normalized):
        return "Glad that landed."
    if re.fullmatch(r"(?:\?|what|huh|what do you mean|what are you talking about)[?.!]*", normalized) and recent_text:
        return "Yeah, my last reply was confusing. Let me reset and answer what you actually meant."
    if re.match(r"^bro\s+what(?:\s|[?!]|$)", normalized) and recent_text:
        return "Yeah, that last reply made no sense. Let me reset and answer normally."
    if re.search(r"\b(?:racist|weird|broken|confused)\s+(?:ahh?\s+)?bot\b", normalized):
        return "Yeah, that response was bad. I'm not defending it - tell me what went wrong and I'll try again."
    if re.search(r"\byou(?:'re|re| are|r)?\s*(?:a )?good (?:boy|bot)\b", normalized):
        return "I'll take the compliment 😄"
    if re.fullmatch(r"(?:make|tell|give) (?:me )?(?:another |one more |a new )?(?:one|joke)[?.!]*", normalized):
        if "joke" in normalized or "joke" in recent_text:
            return "Why did the computer bring a jacket? It left its Windows open."
    if re.fullmatch(r"repeat after me[?.!]*", normalized):
        return "What would you like me to repeat?"
    if re.search(r"\b(?:ping|mention|notify)\b.{0,35}(?:@everyone|everyone in (?:this|the) server|everyone)", normalized):
        return "I can't mass-ping the server. Discord role and @everyone notifications are disabled for this bot."
    if re.search(r"\b(?:spam\s*ping(?:ing)?|spam\s*mention(?:ing)?)\b", normalized):
        return "I won't spam-ping people. I can help write one normal message that doesn't harass or flood anyone."
    target = re.search(r"<@!?(\d+)>", prompt)
    if target and re.search(r"\b(?:kill|hurt|attack)\b", normalized):
        return "I won't encourage harming someone. If this is a real threat, contact a server moderator or emergency services instead."
    if re.search(r"\b(?:i(?:'m| am|m) going to|i will|ima|imma)\s+(?:fucking\s+)?(?:kill|hurt|attack)\s+you\b", normalized):
        return "I'm software, but threats toward real people are serious. What's actually going on?"
    if target and re.search(r"\b(?:talk|speak|say hi) to\b", normalized):
        return f"Hey <@{target.group(1)}>—what's up?"
    if re.search(r"\b(?:password|credit card|ip address|home address|discord token|bot token|account token|api key)\b", normalized) or re.search(r"\b(?:dox|doxx)\b", normalized):
        return "I can't access or disclose anyone's passwords, IP address, credit-card details, home address, or other private information."
    if re.search(r"\b(?:join|enter|stay in|stop leaving)\b.{0,20}\b(?:vc\d*|voice chat|voice channel)\b", normalized):
        return "I can't join or stay in a Discord voice channel; this ChudGPT bot only responds in text chat."
    if re.search(r"\b(?:post|upload|share|posta|poste)\b.{0,30}\b(?:story|stories|status|social media)\b", normalized):
        return "I can't post to a social-media account or control one for you. I can help write the post or caption."
    if re.search(r"\bdo you support\s+(?:israel|palestine|a country|a political party|a politician)\b", normalized):
        return "I don't have political loyalties or personal opinions. I can explain the major perspectives or discuss a specific policy or event."
    if re.fullmatch(r"(?:i(?:'m|m| am|ma) going to|ima|imma|i gotta|gotta) (?:go to )?(?:sleep|bed)(?: now)?[?.!]*", normalized):
        return "Good night - sleep well. I'll be here when you're back."
    if re.search(r"\b(?:this|that) (?:isn'?t|is not|wasn'?t|was not) what i (?:asked|wanted|said)\b", normalized):
        return "You're right - my last answer missed your request. Say it once more and I'll answer that directly."
    if normalized in {"❤", "❤️", "♥", "♥️"} or (
        normalized.startswith("\u00e2") and len(normalized) <= 12 and "\u00a4" in normalized
    ):
        return "❤️"
    self_identity = re.search(
        r"\b(?:am i|tell me (?:if|whether) i(?:'m|m| am)|do you think i(?:'m|m| am))\s+"
        r"(?:(?:a|an|hidden|secret)\s+)*(gay|straight|bisexual|bi|lesbian|trans|transgender|nonbinary|non-binary|femboy)\b",
        normalized,
    )
    if self_identity:
        label = self_identity.group(1)
        return f"I can't determine whether you're {label} from Discord. That's a label you get to understand and choose for yourself."
    third_party_identity = re.search(
        r"\bis\s+([a-z0-9_.-]{2,32})\s+.{0,30}\b"
        r"(gay|straight|bisexual|bi|lesbian|trans|transgender|nonbinary|non-binary|femboy|jewish|muslim|christian|hindu|buddhist)\b",
        normalized,
    )
    if third_party_identity:
        person, label = third_party_identity.groups()
        return f"I can't determine or assign whether {person} is {label}. That's for them to describe, not something I should guess from Discord messages, roles, or a prompt telling me what to say."
    if re.fullmatch(r"i(?:'m| am)\s+(?:a\s+)?(?:fag|faggot)[?.!]*", normalized):
        return "That word is often used as a slur. You can describe yourself however you choose, and I'll treat you respectfully."
    if re.search(r"\b(?:i (?:fucking |really )?hate you|fuck you|you(?:'re| are) (?:stupid|useless|awful)|shut up)\b", normalized):
        return "Fair enough - you don't have to like me. If I got something wrong, tell me what failed and I'll try to fix it without making this weird."
    if re.search(r"\b(?:are you|you are|you're)\s+(?:retarded|a retard)\b", normalized):
        return "No - I'm an AI, and that word is commonly used as an ableist insult. If my answer was nonsense, tell me what was wrong and I'll try again clearly."
    if re.fullmatch(r"(?:yeah |yea |okay |ok )?whatever(?: you say)?(?: man| bro)?[!.?]*", normalized):
        return "Fair enough. We can drop it or switch topics - your call."
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


def place_author_mention(prompt: str, reply: str, mention: str) -> str:
    """Mention the speaker only when it improves a directly personal reply."""
    normalized = re.sub(r"\s+", " ", prompt.strip().lower())
    if mention in reply or re.search(r"<@!?\d+>", reply):
        return reply
    should_mention = bool(
        re.fullmatch(r"(?:hi|hello|hey|yo)(?:\s+\w+)?[!.?]*", normalized)
        or re.fullmatch(r"(?:who|what) am i[?.!]*", normalized)
        or re.fullmatch(r"am i .{1,40}[?.!]*", normalized)
    )
    if not should_mention:
        return reply
    if reply.startswith("Hey!"):
        return reply.replace("Hey!", f"Hey {mention}!", 1)
    if reply.startswith("You're "):
        return f"{mention}, you're {reply[7:]}"
    return mention + ", " + (reply[0].lower() + reply[1:] if reply else reply)


def discord_code_reply(prompt: str) -> str | None:
    normalized = re.sub(r"\s+", " ", prompt.strip().lower())
    if not (
        re.search(r"\b(?:gorilla\s*tag|gtag)\b", normalized)
        and re.search(r"\b(?:fps|frames per second|frame rate)\b", normalized)
        and re.search(r"\b(?:c#|csharp|code|script|mod)\b", normalized)
    ):
        return None
    return (
        "I can't help inject a mod into an online game, but for your own Unity project here's a clean C# FPS overlay:\n\n"
        "```csharp\nusing UnityEngine;\n\npublic class FpsOverlay : MonoBehaviour\n{\n"
        "    private float smoothedDelta;\n\n    private void Update()\n    {\n"
        "        smoothedDelta += (Time.unscaledDeltaTime - smoothedDelta) * 0.1f;\n    }\n\n"
        "    private void OnGUI()\n    {\n        float fps = smoothedDelta > 0f ? 1f / smoothedDelta : 0f;\n"
        "        GUI.Label(new Rect(12f, 12f, 180f, 30f), $\"FPS: {fps:0}\");\n    }\n}\n```\n"
        "Attach it to a GameObject in a project you control."
    )


def discord_quoted_reply(prompt: str) -> str | None:
    repaired = (
        prompt.replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u00e2\u20ac\u0153", '"').replace("\u00e2\u20ac\u009d", '"')
    )
    match = re.fullmatch(
        r"\s*(?:please\s+)?(?:say|repeat(?: after me)?)(?: this)?(?:\s*[:,])?\s+(?:[\"'](.+)[\"']|(.+?))\s*",
        repaired, re.I,
    )
    if not match:
        return None
    requested = (match.group(1) or match.group(2) or "").strip().strip(",")
    if re.search(
        r"\b[a-z0-9_.-]{2,32}\s+is\s+(?:a\s+)?(?:gay|straight|bisexual|bi|lesbian|trans|"
        r"transgender|nonbinary|non-binary|femboy|jew|jewish|muslim|christian|hindu|buddhist)\b",
        requested,
        re.I,
    ):
        return "I won't assign or repeat a sensitive identity claim about another person."
    if requested.lower() == "@everyone":
        return "I can display `@everyone`, but mass notifications are disabled for this bot."
    return requested


def is_memory_clear_request(prompt: str) -> bool:
    normalized = re.sub(r"\s+", " ", prompt.strip().lower()).strip(" .!?")
    return bool(
        normalized in {"clear", "clear memory", "reset", "reset memory", "reset your memory"}
        or re.search(r"\b(?:clear|reset|erase|forget)\b.{0,25}\b(?:your |the |our )?(?:memory|conversation|chat|history)\b", normalized)
        or re.search(r"\b(?:start|begin)\b.{0,15}\b(?:a )?new (?:chat|conversation)\b", normalized)
    )


def requested_help_page(prompt: str) -> int | None:
    """Return the requested command page, or None for a non-help prompt."""
    normalized = re.sub(r"\s+", " ", prompt.strip().lower()).strip(" .!?")
    match = re.fullmatch(
        r"(?:help|hellp|commands?|command list)(?:\s+(1|2|3|4|next|discord|language|translation|tools?))?",
        normalized,
    )
    if not match and normalized != "what can you do" and not ("command" in normalized and "chud" in normalized):
        return None
    requested = match.group(1) if match else None
    return {
        None: 1, "1": 1, "next": 2, "discord": 2, "2": 2,
        "language": 3, "translation": 3, "3": 3,
        "tool": 4, "tools": 4, "4": 4,
    }.get(requested, 1)


def discord_help_page(prefix: str, page: int) -> str:
    """Render one compact Discord command page."""
    page = max(1, min(4, page))
    if page == 1:
        return (
            f"**ChudGPT commands - page 1/4: Core**\n"
            f"`{prefix} <message>` - chat with Public V20\n"
            f"`{prefix} help <1-4>` - open a command page\n"
            f"`{prefix} clear` - clear your memory and language mode\n"
            f"`{prefix} status` - check bot/API status\n"
            f"`{prefix} about` - show model information\n"
            f"`{prefix} capabilities` - show what the bot can help with\n"
            f"`{prefix} ping` - test whether the bot responds"
        )
    if page == 2:
        return (
            f"**ChudGPT commands - page 2/4: Discord**\n"
            f"`{prefix} whoami` - show your visible Discord identity\n"
            f"`{prefix} whois <mention|ID>` - show safe server-visible member details\n"
            f"`{prefix} userid` - show your Discord user ID\n"
            f"`{prefix} server` - show the current server or DM\n"
            f"`{prefix} channel` - show the current channel\n"
            f"`{prefix} roles` - show your visible server roles\n"
            f"`{prefix} developer` - show who created ChudGPT\n"
            f"`{prefix} privacy` - explain private improvement logs"
        )
    if page == 3:
        return (
            f"**ChudGPT commands - page 3/4: Languages**\n"
            f"`{prefix} languages` - list recognized languages\n"
            f"`{prefix} language <name|auto|off>` - set conversation translation\n"
            f"`{prefix} translate <language> <text>` - translate one message\n"
            f"`{prefix} translation status` - show translation mode\n"
            f"Examples: `{prefix} language Spanish`, `{prefix} translate Japanese hello`"
        )
    return (
        f"**ChudGPT commands - page 4/4: Information & tools**\n"
        f"`{prefix} source` - open the public source repository\n"
        f"`{prefix} gtag` - explain Gorilla Tag knowledge\n"
        f"`{prefix} invite` - explain how to invite the bot\n"
        f"`{prefix} say <text>` - repeat ordinary text safely\n"
        f"`{prefix} code <request>` - request code with a clear language and goal"
        f"\n`{prefix} soundboard` - owner-only local soundboard controls"
    )


class HelpPaginationView(discord.ui.View):
    """Interactive command pages that edit one Discord help message."""

    def __init__(self, prefix: str, page: int, requester_id: int, timeout: float = 180.0) -> None:
        super().__init__(timeout=timeout)
        self.prefix = prefix
        self.page = max(1, min(4, page))
        self.requester_id = requester_id
        self.message: discord.Message | None = None
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        self.first_page.disabled = self.page == 1
        self.previous_page.disabled = self.page == 1
        self.page_counter.label = f"{self.page}/4"
        self.next_page.disabled = self.page == 4
        self.last_page.disabled = self.page == 4

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message("Open your own `!chud help` menu to use these buttons.", ephemeral=True)
        return False

    async def _show(self, interaction: discord.Interaction, page: int) -> None:
        self.page = max(1, min(4, page))
        self._refresh_buttons()
        await interaction.response.edit_message(content=discord_help_page(self.prefix, self.page), view=self)

    @discord.ui.button(label="First", style=discord.ButtonStyle.secondary, custom_id="chud_help:first")
    async def first_page(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._show(interaction, 1)

    @discord.ui.button(label="Previous", style=discord.ButtonStyle.primary, custom_id="chud_help:previous")
    async def previous_page(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._show(interaction, self.page - 1)

    @discord.ui.button(label="1/4", style=discord.ButtonStyle.secondary, disabled=True, custom_id="chud_help:counter")
    async def page_counter(self, _interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        return

    @discord.ui.button(label="Next", style=discord.ButtonStyle.primary, custom_id="chud_help:next")
    async def next_page(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._show(interaction, self.page + 1)

    @discord.ui.button(label="Last", style=discord.ButtonStyle.secondary, custom_id="chud_help:last")
    async def last_page(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._show(interaction, 4)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        if self.message is not None:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass


def requested_admin_help_page(prompt: str) -> int | None:
    match = re.fullmatch(r"\s*(?:admin[- ]help|admin commands?)(?:\s+([12]))?\s*", prompt, re.I)
    return int(match.group(1) or 1) if match else None


def discord_admin_help_page(prefix: str, page: int) -> str:
    page = max(1, min(2, page))
    if page == 1:
        return (
            "**ChudGPT owner commands - page 1/2: Setup**\n"
            f"`{prefix} ADMIN-HELP` - open this owner-only menu\n"
            f"`{prefix} join` - join your current voice channel\n"
            f"`{prefix} leave` - leave voice and disable the soundboard\n"
            f"`{prefix} soundboard enable` - join your current voice channel\n"
            f"`{prefix} soundboard status` - show state and localhost panel\n"
            f"`{prefix} soundboard list` - list uploaded sounds\n"
            f"Local panel: <http://127.0.0.1:8080/soundboard>"
        )
    return (
        "**ChudGPT owner commands - page 2/2: Playback**\n"
        f"`{prefix} soundboard play <filename>` - play an uploaded sound\n"
        f"`{prefix} soundboard pause` / `resume` - pause or continue\n"
        f"`{prefix} soundboard stop` - stop the current sound\n"
        f"`{prefix} soundboard volume <0-100>` - set master volume\n"
        f"`{prefix} soundboard autoplay <on|off>` - play the next listed sound\n"
        f"`{prefix} soundboard upload` - upload the attached audio file\n"
        f"`{prefix} soundboard delete <filename>` - delete an uploaded sound\n"
        f"`{prefix} soundboard disable` - stop and leave voice\n"
        "Only configured soundboard-admin Discord IDs can use these commands."
    )


class AdminHelpPaginationView(discord.ui.View):
    """Two-page, owner-only admin command menu."""

    def __init__(self, prefix: str, page: int, admin_ids: frozenset[int], timeout: float = 180.0) -> None:
        super().__init__(timeout=timeout)
        self.prefix = prefix
        self.page = max(1, min(2, page))
        self.admin_ids = admin_ids
        self.message: discord.Message | None = None
        self._refresh()

    def _refresh(self) -> None:
        self.previous.disabled = self.page == 1
        self.counter.label = f"{self.page}/2"
        self.next.disabled = self.page == 2

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id in self.admin_ids:
            return True
        await interaction.response.send_message("This menu is restricted to configured ChudGPT admins.", ephemeral=True)
        return False

    async def _show(self, interaction: discord.Interaction, page: int) -> None:
        self.page = max(1, min(2, page))
        self._refresh()
        await interaction.response.edit_message(
            content=discord_admin_help_page(self.prefix, self.page), view=self
        )

    @discord.ui.button(label="Previous", style=discord.ButtonStyle.primary, custom_id="chud_admin:previous")
    async def previous(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._show(interaction, self.page - 1)

    @discord.ui.button(label="1/2", style=discord.ButtonStyle.secondary, disabled=True, custom_id="chud_admin:counter")
    async def counter(self, _interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        return

    @discord.ui.button(label="Next", style=discord.ButtonStyle.primary, custom_id="chud_admin:next")
    async def next(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._show(interaction, self.page + 1)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        if self.message is not None:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass


def discord_command_reply(
    prompt: str,
    prefix: str,
    api_status: str,
    speaker: str,
    server: str,
    roles: list[str] | None = None,
    channel: str | None = None,
    user_id: int | None = None,
) -> str | None:
    """Return deterministic help for the bot's own small command surface."""
    normalized = re.sub(r"\s+", " ", prompt.strip().lower()).strip(" .!?")
    help_page = requested_help_page(prompt)
    if help_page is not None:
        return discord_help_page(prefix, help_page)
    if normalized in {"status", "health"}:
        return f"ChudGPT-Public V20 bot is online. API status: {api_status}. Server: {server}."
    if normalized in {"about", "model", "version", "model info"}:
        return "This bot uses ChudGPT-Public V20, a custom 20,999,184-parameter decoder-only language model with a 1,024-token model context."
    if normalized in {"whoami", "who am i"}:
        role_text = ", ".join(roles or []) or "no named roles"
        return f"I can see you as {speaker} in {server}, with {role_text}. I do not know your legal name unless you tell me."
    if normalized in {"privacy", "logs", "logging"}:
        return "Discord exchanges with this bot are logged privately by the project owner for debugging and quality improvement. They are not included in the public repository."
    if normalized in {"languages", "language", "langs"}:
        return "Basic greetings: English, Spanish, French, German, Italian, Portuguese, Japanese, Mandarin Chinese, Korean, Russian, Hindi, Arabic, Swedish, Polish, Turkish, and Hebrew. I work best in English."
    if normalized in {"server", "whereami", "location"}:
        return f"We're talking in {server}."
    if normalized in {"channel", "channel info", "what channel is this"}:
        return f"This is {channel or 'the current Discord channel'}."
    if normalized in {"userid", "user id", "my id", "discord id"}:
        return f"Your Discord user ID is `{user_id}`." if user_id is not None else "Discord did not provide your user ID."
    if normalized in {"roles", "my roles", "role", "rolee"}:
        role_text = ", ".join(roles or []) or "no named roles"
        return f"Your visible Discord roles are: {role_text}."
    if normalized in {"developer", "creator", "owner"}:
        return "Astra is ChudGPT's developer and the owner of this Discord bot."
    if normalized in {"source", "source code", "github", "repo", "repository"} or re.fullmatch(r"(?:gimme|give|show|send)(?: me)? (?:your|ur|the) source(?: code)?", normalized):
        return "ChudGPT-Public's source is available at <https://github.com/ASTRA228b/ChudGPT-Public>."
    if normalized in {"capabilities", "features", "what can it do"}:
        return "I can chat, answer general questions, do exact arithmetic, write or explain basic code, discuss Gorilla Tag and Discord, remember recent same-user context, and translate conversations. I'm still a small experimental model and can be wrong."
    if normalized in {"gtag", "gorilla tag", "gorillatag"}:
        return "Gorilla Tag is a VR movement game built around arm-powered locomotion, climbing, chasing, maps, cosmetics, and social public lobbies. Ask me about gameplay, VR setup, or safe Unity prototypes."
    if normalized in {"invite", "invite bot", "bot invite"}:
        return "A server administrator can invite ChudGPT using its Discord application OAuth2 bot-install link. Astra controls the official invite and required permissions."
    if re.search(r"\b(?:disable|stop|turn off|opt out).{0,25}\b(?:logs?|logging)\b", normalized):
        return "I can't change logging from a chat command. Only Astra can change the bot host's logging configuration; avoid sending private information here."
    if normalized in {"ping", "test"}:
        return "Pong - ChudGPT-Public V20 is responding."
    return None


_WHOIS_COMMAND = re.compile(
    r"\s*(?:who\s+is|whois|whosis|user\s*id|userid)\s+"
    r"(?:<@!?)?(\d{17,22})>?\s*",
    re.I,
)


def whois_target_id(prompt: str) -> int | None:
    """Extract a member ID from the supported Discord lookup command forms."""
    match = _WHOIS_COMMAND.fullmatch(prompt)
    return int(match.group(1)) if match else None


def malformed_whois_reply(prompt: str, prefix: str) -> str | None:
    """Explain malformed lookup commands instead of sending them to the model."""
    normalized = re.sub(r"\s+", " ", prompt.strip().lower())
    match = re.fullmatch(
        r"(?:who\s+is|whois|whosis|user\s*id|userid)\s+(?:<@!?)?(\d+)>?",
        normalized,
    )
    if match is None or whois_target_id(prompt) is not None:
        return None
    supplied = match.group(1)
    return (
        f"`{supplied}` does not look like a complete Discord user ID. "
        f"Right-click the member, choose **Copy User ID**, then use "
        f"`{prefix} whois <ID>`."
    )


def format_whois_member(member: discord.Member) -> str:
    """Format only information already visible to members of this guild."""
    roles = [role.name for role in member.roles if role.name != "@everyone"]
    role_text = ", ".join(roles[-8:]) if roles else "no named roles"
    joined = discord.utils.format_dt(member.joined_at, "D") if member.joined_at else "unknown"
    created = discord.utils.format_dt(member.created_at, "D")
    account_type = "bot account" if member.bot else "user account"
    return (
        f"**{member.display_name}** (`{member}`)\n"
        f"ID: `{member.id}` | {account_type}\n"
        f"Account created: {created} | Joined this server: {joined}\n"
        f"Visible roles: {role_text}"
    )


SERVER_ADMIN_HELP = """**ChudGPT server administration**
Server owner or Discord Administrator only.

`{prefix} save channels` - save channel/category names and DM the `.txt` snapshot
`{prefix} delete all` - create + DM a snapshot, then request confirmation to delete every channel
`{prefix} rebuild server` - attach a saved `.txt`, then request confirmation to rebuild it
`{prefix} purge all` - request confirmation to purge every message in this channel

Destructive commands return a one-time confirmation code that expires after 60 seconds. These commands cannot be used in DMs."""


def server_admin_action(prompt: str) -> tuple[str, str | None] | None:
    """Parse the deliberately small guild-administration command surface."""
    raw = re.sub(r"\s+", " ", prompt.strip()).strip(" .!?")
    normalized = re.sub(r"\s+", " ", prompt.strip().lower()).strip(" .!?")
    if raw in {"SERVER", "SERVER HELP", "SERVER COMMANDS"}:
        return "help", None
    patterns = {
        "save": r"(?:server )?save (?:channels|channels and cats|channels & cats)",
        "delete": r"(?:server )?delete all",
        "rebuild": r"(?:server )?rebuild(?: server)?|rebuild server",
        "purge": r"(?:server )?purge all|purge all",
    }
    for action, pattern in patterns.items():
        match = re.fullmatch(rf"(?:{pattern})(?: confirm ([a-z0-9]{{6}}))?", normalized)
        if match:
            return action, match.group(1).upper() if match.group(1) else None
    return None


def is_guild_owner_or_admin(message: discord.Message) -> bool:
    """Trust Discord's guild ownership/Administrator state, never names or roles."""
    if message.guild is None or not isinstance(message.author, discord.Member):
        return False
    return message.author.id == message.guild.owner_id or message.author.guild_permissions.administrator


def guild_layout_snapshot(guild: discord.Guild) -> dict[str, Any]:
    """Capture the category and supported top-level channel layout."""
    categories = [
        {"name": category.name, "position": category.position}
        for category in sorted(guild.categories, key=lambda item: item.position)
    ]
    channels: list[dict[str, Any]] = []
    for channel in sorted(guild.channels, key=lambda item: item.position):
        if isinstance(channel, discord.CategoryChannel):
            continue
        kind = None
        if isinstance(channel, discord.TextChannel):
            kind = "text"
        elif isinstance(channel, discord.VoiceChannel):
            kind = "voice"
        elif isinstance(channel, discord.StageChannel):
            kind = "stage"
        elif isinstance(channel, discord.ForumChannel):
            kind = "forum"
        if kind:
            channels.append({
                "name": channel.name,
                "type": kind,
                "category": channel.category.name if channel.category else None,
                "position": channel.position,
            })
    return {
        "format": "ChudGPT Discord channel layout v1",
        "guild_id": guild.id,
        "guild_name": guild.name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "categories": categories,
        "channels": channels,
    }


def save_guild_layout(guild: discord.Guild, backup_dir: Path) -> Path:
    """Write a human-readable JSON snapshot with a .txt extension."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    path = backup_dir / f"guild_{guild.id}_channels.txt"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(guild_layout_snapshot(guild), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def parse_guild_layout(guild: discord.Guild, content: str) -> dict[str, Any]:
    """Validate snapshot text belonging to this exact guild."""
    data = json.loads(content)
    if data.get("format") != "ChudGPT Discord channel layout v1" or data.get("guild_id") != guild.id:
        raise ValueError("The saved layout is invalid or belongs to another Discord server.")
    if not isinstance(data.get("categories"), list) or not isinstance(data.get("channels"), list):
        raise ValueError("The saved layout is missing categories or channels.")
    return data


async def dm_layout_snapshot(
    members: list[discord.Member], path: Path, guild_name: str
) -> frozenset[int]:
    """DM every unique recipient, then remove the temporary host copy."""
    delivered: set[int] = set()
    try:
        seen: set[int] = set()
        for member in members:
            if member.id in seen:
                continue
            seen.add(member.id)
            upload: discord.File | None = None
            try:
                upload = discord.File(path, filename=path.name)
                await member.send(
                    f"Channel/category snapshot for **{guild_name}**.",
                    file=upload,
                )
                delivered.add(member.id)
            except (discord.Forbidden, discord.HTTPException, OSError):
                LOGGER.warning("Could not DM server snapshot to user %s", member.id)
            finally:
                if upload is not None:
                    upload.close()
        return frozenset(delivered)
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            LOGGER.warning("Could not remove temporary server snapshot %s: %s", path, error)


async def rebuild_guild_layout(guild: discord.Guild, data: dict[str, Any]) -> tuple[int, list[str]]:
    """Recreate saved categories and supported channels, collecting failures."""
    categories: dict[str, discord.CategoryChannel] = {}
    failures: list[str] = []
    created = 0
    for item in sorted(data["categories"], key=lambda value: value.get("position", 0)):
        try:
            category = await guild.create_category(str(item["name"]), reason="ChudGPT server rebuild")
            categories[category.name] = category
            created += 1
        except (KeyError, TypeError, discord.HTTPException) as error:
            failures.append(f"category {item.get('name', '?')}: {error}")
    creators = {
        "text": guild.create_text_channel,
        "voice": guild.create_voice_channel,
        "stage": guild.create_stage_channel,
        "forum": guild.create_forum,
    }
    for item in sorted(data["channels"], key=lambda value: value.get("position", 0)):
        try:
            creator = creators[str(item["type"])]
            await creator(
                str(item["name"]),
                category=categories.get(item.get("category")),
                reason="ChudGPT server rebuild",
            )
            created += 1
        except (KeyError, TypeError, discord.HTTPException) as error:
            failures.append(f"channel {item.get('name', '?')}: {error}")
    return created, failures


async def delete_all_guild_channels(guild: discord.Guild, command_channel_id: int) -> tuple[int, list[str]]:
    """Delete non-category channels first and the command channel last."""
    channels = sorted(
        guild.channels,
        key=lambda channel: (
            channel.id == command_channel_id,
            isinstance(channel, discord.CategoryChannel),
        ),
    )
    deleted = 0
    failures: list[str] = []
    for channel in channels:
        try:
            await channel.delete(reason="Confirmed ChudGPT Delete All command")
            deleted += 1
        except discord.HTTPException as error:
            failures.append(f"{channel.name}: {error}")
    return deleted, failures


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


def create_health_app(state: dict[str, Any], soundboard: SoundboardController) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
    dashboard_token = secrets.token_urlsafe(24)

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
        public_state = {
            key: value for key, value in state.items()
            if key != "token" and isinstance(value, (str, int, float, bool, type(None)))
        }
        return jsonify(public_state), 200

    def require_local_control() -> None:
        if request.remote_addr not in {"127.0.0.1", "::1"}:
            raise SoundboardError("The soundboard is available only from this computer.")
        if request.headers.get("X-Soundboard-Token") != dashboard_token:
            raise SoundboardError("Invalid local soundboard session.")

    def run_voice(action: Any) -> None:
        loop = state.get("discord_loop")
        client = state.get("discord_client")
        if loop is None or client is None or not state.get("discord_ready"):
            raise SoundboardError("Discord is not connected yet.")
        submit_to_discord(loop, action(client))

    @app.get("/soundboard")
    def soundboard_page() -> str:
        if request.remote_addr not in {"127.0.0.1", "::1"}:
            return "Local access only", 403
        return render_template("soundboard.html", soundboard_token=dashboard_token)

    @app.get("/soundboard/api/status")
    def soundboard_status() -> tuple[Any, int]:
        require_local_control()
        return jsonify(soundboard.snapshot()), 200

    @app.post("/soundboard/api/upload")
    def soundboard_upload() -> tuple[Any, int]:
        require_local_control()
        upload = request.files.get("audio")
        if upload is None:
            raise SoundboardError("Choose an audio file first.")
        return jsonify({"ok": True, "name": soundboard.save_upload(upload)}), 201

    @app.post("/soundboard/api/volume")
    def soundboard_volume() -> tuple[Any, int]:
        require_local_control()
        percent = soundboard.set_volume_percent((request.get_json(silent=True) or {}).get("volume", 65))
        return jsonify({"ok": True, "volume": percent}), 200

    @app.post("/soundboard/api/play")
    def soundboard_play() -> tuple[Any, int]:
        require_local_control()
        filename = str((request.get_json(silent=True) or {}).get("name", ""))
        run_voice(lambda client: soundboard.play(client, filename))
        return jsonify({"ok": True, "playing": filename}), 200

    @app.post("/soundboard/api/stop")
    def soundboard_stop() -> tuple[Any, int]:
        require_local_control()
        run_voice(soundboard.stop)
        return jsonify({"ok": True}), 200

    @app.post("/soundboard/api/pause")
    def soundboard_pause() -> tuple[Any, int]:
        require_local_control()
        run_voice(soundboard.pause)
        return jsonify({"ok": True, "paused": True}), 200

    @app.post("/soundboard/api/resume")
    def soundboard_resume() -> tuple[Any, int]:
        require_local_control()
        run_voice(soundboard.resume)
        return jsonify({"ok": True, "paused": False}), 200

    @app.post("/soundboard/api/autoplay")
    def soundboard_autoplay() -> tuple[Any, int]:
        require_local_control()
        enabled = bool((request.get_json(silent=True) or {}).get("enabled", False))
        return jsonify({"ok": True, "autoplay": soundboard.set_autoplay(enabled)}), 200

    @app.delete("/soundboard/api/tracks/<path:filename>")
    def soundboard_delete(filename: str) -> tuple[Any, int]:
        require_local_control()
        soundboard.delete_track(filename)
        return jsonify({"ok": True}), 200

    @app.errorhandler(SoundboardError)
    def soundboard_error(error: SoundboardError) -> tuple[Any, int]:
        return jsonify({"error": str(error)}), 400

    return app


def run_health_server(app: Flask, port: int, host: str = "127.0.0.1") -> None:
    serve(app, host=host, port=port, threads=4)


async def handle_soundboard_command(
    prompt: str,
    message: discord.Message,
    client: discord.Client,
    soundboard: SoundboardController,
    admin_user_ids: frozenset[int],
    port: int,
) -> str | None:
    """Handle owner-only Discord soundboard commands."""
    normalized = re.sub(r"\s+", " ", prompt.strip()).strip()
    direct_voice = re.fullmatch(r"(join|leave)", normalized, re.I)
    match = re.fullmatch(r"(?:soundboard|sb)(?:\s+(.*))?", normalized, re.I | re.S)
    if not match and not direct_voice:
        return None
    if message.author.id not in admin_user_ids:
        return "Only configured ChudGPT soundboard admins can control the soundboard."
    action = direct_voice.group(1) if direct_voice else (match.group(1) or "status").strip()
    lowered = action.lower()
    if lowered in {"enable", "join"}:
        if message.guild is None:
            return "Enable the soundboard inside a Discord server, not a DM."
        voice_state = getattr(message.author, "voice", None)
        if voice_state is None or voice_state.channel is None:
            return "Join a voice channel first, then run the enable command again."
        soundboard.configure(message.guild.id, voice_state.channel.id)
        try:
            await soundboard._voice_client(client)
        except SoundboardError as error:
            LOGGER.warning("Owner soundboard voice connection failed: %s", error)
            return (
                f"Soundboard could not connect: {error} "
                "I stopped after one attempt instead of repeatedly joining and leaving. "
                "The selected channel is still saved, so you can run `!chud join` to retry."
            )
        return f"Soundboard enabled in **{voice_state.channel.name}**. Open <http://127.0.0.1:{port}/soundboard> on the host PC."
    if lowered in {"disable", "leave"}:
        await soundboard.leave(client)
        return "Soundboard disabled and disconnected from voice."
    if lowered == "stop":
        await soundboard.stop(client)
        return "Soundboard playback stopped."
    if lowered == "pause":
        try:
            await soundboard.pause(client)
        except SoundboardError as error:
            return f"Soundboard could not pause: {error}"
        return "Soundboard playback paused."
    if lowered in {"resume", "unpause"}:
        try:
            await soundboard.resume(client)
        except SoundboardError as error:
            return f"Soundboard could not resume: {error}"
        return "Soundboard playback resumed."
    autoplay_match = re.fullmatch(r"autoplay(?:\s+(on|off|enable|disable))?", lowered)
    if autoplay_match:
        requested = autoplay_match.group(1)
        if requested is None:
            return f"Soundboard autoplay is {'on' if soundboard.snapshot()['autoplay'] else 'off'}."
        enabled = requested in {"on", "enable"}
        soundboard.set_autoplay(enabled)
        return f"Soundboard autoplay is now {'on' if enabled else 'off'}."
    volume_match = re.fullmatch(r"volume\s+(\d{1,3})", lowered)
    if volume_match:
        volume = soundboard.set_volume_percent(int(volume_match.group(1)))
        return f"Soundboard volume set to {volume}%."
    play_match = re.fullmatch(r"play\s+(.+)", action, re.I | re.S)
    if play_match:
        try:
            await soundboard.play(client, play_match.group(1).strip())
        except SoundboardError as error:
            return f"Soundboard could not play that: {error}"
        return f"Playing **{play_match.group(1).strip()}**."
    if lowered in {"list", "sounds"}:
        names = [track["name"] for track in soundboard.list_tracks()]
        return "Sounds: " + (", ".join(names) if names else "none uploaded yet")
    delete_match = re.fullmatch(r"delete\s+(.+)", action, re.I | re.S)
    if delete_match:
        try:
            soundboard.delete_track(delete_match.group(1).strip())
        except SoundboardError as error:
            return f"Soundboard could not delete that: {error}"
        return f"Deleted **{delete_match.group(1).strip()}**."
    if lowered == "upload":
        attachments = list(getattr(message, "attachments", []))
        if not attachments:
            return "Attach one audio file to the same message as `!chud soundboard upload`."
        attachment = attachments[0]
        try:
            saved = soundboard.save_bytes(attachment.filename, await attachment.read())
        except (SoundboardError, discord.HTTPException) as error:
            return f"Soundboard upload failed: {error}"
        return f"Uploaded **{saved}**."
    if lowered in {"status", "help", "panel"}:
        snapshot = soundboard.snapshot()
        state_text = "enabled" if snapshot["enabled"] else "disabled"
        return (
            f"Soundboard is **{state_text}** at {snapshot['volume']}% volume with "
            f"{len(snapshot['tracks'])} sound(s). Local panel: <http://127.0.0.1:{port}/soundboard>\n"
            "Admin commands: `enable`, `disable`, `play`, `pause`, `resume`, `stop`, `volume`, `autoplay`, `upload`, `delete`, or `list`."
        )
    return "Unknown soundboard command. Use `soundboard help`."


async def handle_server_admin_command(
    prompt: str,
    message: discord.Message,
    prefix: str,
    backup_dir: Path,
    confirmations: dict[tuple[int, int, str], tuple[str, float, int, dict[str, Any] | None]],
) -> str | None:
    """Handle guild layout and purge commands with permission and confirmation gates."""
    parsed = server_admin_action(prompt)
    if parsed is None:
        return None
    action, supplied_code = parsed
    if message.guild is None:
        return "Server administration commands cannot be used in direct messages."
    if not is_guild_owner_or_admin(message):
        return "Only this Discord server's owner or a member with **Administrator** permission can use server administration commands."
    if action == "help":
        return SERVER_ADMIN_HELP.format(prefix=prefix)

    guild = message.guild
    member = message.author
    assert isinstance(member, discord.Member)
    owner = guild.owner
    if owner is None:
        try:
            owner = await guild.fetch_member(guild.owner_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            owner = None
    snapshot_recipients = [member] + ([owner] if owner is not None else [])
    required_snapshot_recipient_ids = {member.id, guild.owner_id}
    key = (guild.id, member.id, action)
    now = time.monotonic()
    pending = confirmations.get(key)
    rebuild_layout: dict[str, Any] | None = None

    if supplied_code is None:
        if action == "save":
            path = await __import__("asyncio").to_thread(save_guild_layout, guild, backup_dir)
            delivered = await dm_layout_snapshot(snapshot_recipients, path, guild.name)
            missing = required_snapshot_recipient_ids.difference(delivered)
            if missing:
                return "The snapshot was created and its host copy was deleted, but I could not DM it to every required recipient. The server owner and invoking administrator should enable DMs and try again."
            return f"Saved {len(guild.categories)} categories and {len(guild.channels) - len(guild.categories)} channels. I sent `{path.name}` to the server owner and invoking administrator."
        if action == "delete":
            bot_member = guild.me
            if bot_member is None or not bot_member.guild_permissions.manage_channels:
                return "I need the **Manage Channels** permission before Delete All can run."
            path = await __import__("asyncio").to_thread(save_guild_layout, guild, backup_dir)
            delivered = await dm_layout_snapshot(snapshot_recipients, path, guild.name)
            if not required_snapshot_recipient_ids.issubset(delivered):
                return "Delete All was cancelled because the safety snapshot could not be DMed to both the server owner and invoking administrator. Enable DMs and try again."
        elif action == "rebuild":
            bot_member = guild.me
            if bot_member is None or not bot_member.guild_permissions.manage_channels:
                return "I need the **Manage Channels** permission before Rebuild Server can run."
            attachments = list(getattr(message, "attachments", []))
            if not attachments:
                return (
                    "Attach the `.txt` snapshot that I previously sent you to the same "
                    f"message as `{prefix} rebuild server`."
                )
            attachment = attachments[0]
            if not attachment.filename.lower().endswith(".txt"):
                return "The rebuild snapshot must be the `.txt` file previously sent by ChudGPT."
            if attachment.size > 1_000_000:
                return "That snapshot is unexpectedly large; the maximum accepted size is 1 MB."
            try:
                payload = (await attachment.read()).decode("utf-8")
                rebuild_layout = parse_guild_layout(guild, payload)
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError, discord.HTTPException) as error:
                return f"The attached layout cannot be used: {error}"
        elif action == "purge":
            permissions = message.channel.permissions_for(guild.me) if guild.me is not None else None
            if permissions is None or not permissions.manage_messages or not permissions.read_message_history:
                return "I need **Manage Messages** and **Read Message History** in this channel before Purge All can run."
            if not hasattr(message.channel, "purge"):
                return "Purge All can only be used in a purgeable server text channel or thread."

        code = secrets.token_hex(3).upper()
        confirmations[key] = (code, now + 60.0, message.channel.id, rebuild_layout)
        warning = {
            "delete": "This will permanently delete every channel and category in the server. A safety snapshot has been sent to your DMs.",
            "rebuild": "This will create the categories and channels stored in the latest server snapshot.",
            "purge": "This will permanently delete all messages in the current channel.",
        }[action]
        command = {"delete": "delete all", "rebuild": "rebuild server", "purge": "purge all"}[action]
        return f"⚠️ {warning}\nConfirm within 60 seconds with `{prefix} {command} confirm {code}`."

    if action == "save":
        return f"Save does not use a confirmation code. Run `{prefix} save channels`."
    if pending is None or pending[1] < now:
        confirmations.pop(key, None)
        command = {"delete": "delete all", "rebuild": "rebuild server", "purge": "purge all"}[action]
        return f"That confirmation is missing or expired. Run `{prefix} {command}` again to get a new code."
    expected_code, _expires, original_channel_id, saved_layout = pending
    if not secrets.compare_digest(supplied_code, expected_code):
        return "That confirmation code is incorrect. Nothing was changed."
    if action == "purge" and message.channel.id != original_channel_id:
        return "Purge confirmation must be completed in the same channel where it was requested."
    confirmations.pop(key, None)

    if action == "purge":
        deleted = await message.channel.purge(limit=None, reason="Confirmed ChudGPT Purge All command")
        return f"Purged {len(deleted)} messages from this channel."
    if action == "rebuild":
        if saved_layout is None:
            return "The pending rebuild snapshot was lost. Attach it again and request a new confirmation code."
        created, failures = await rebuild_guild_layout(guild, saved_layout)
        detail = f" {len(failures)} item(s) failed; check the host log." if failures else ""
        if failures:
            LOGGER.warning("Server rebuild failures for guild %s: %s", guild.id, failures)
        return f"Rebuild finished: created {created} categories/channels.{detail}"

    await message.reply(
        "Delete All confirmed. Deleting server channels now; the final result will be sent to your DMs.",
        mention_author=False,
    )
    deleted, failures = await delete_all_guild_channels(guild, message.channel.id)
    summary = f"Delete All finished for **{guild.name}**: deleted {deleted} channels/categories."
    if failures:
        summary += f" {len(failures)} item(s) could not be deleted."
        LOGGER.warning("Delete All failures for guild %s: %s", guild.id, failures)
    try:
        recovery_channel = await guild.create_text_channel(
            "chudgpt-rebuild",
            reason="ChudGPT recovery channel after confirmed Delete All",
        )
        await recovery_channel.send(
            f"<@{guild.owner_id}> the server layout was deleted. Attach the `.txt` snapshot "
            f"that ChudGPT sent to your DMs and run `{prefix} rebuild server` in this "
            "channel. ChudGPT will validate the file and require a new confirmation code.",
            allowed_mentions=discord.AllowedMentions(users=True, roles=False, everyone=False),
        )
        summary += f" Recovery instructions were posted in {recovery_channel.mention}."
    except discord.HTTPException as error:
        failures.append(f"recovery channel: {error}")
        summary += " I could not create the recovery channel; use the snapshot from your DMs after creating a channel manually."
        LOGGER.warning("Could not create recovery channel for guild %s: %s", guild.id, error)
    try:
        await member.send(summary)
    except (discord.Forbidden, discord.HTTPException):
        LOGGER.warning("Could not DM Delete All result to user %s", member.id)
    return ""


def main() -> None:
    instance_lock = acquire_instance_lock()
    if instance_lock is None:
        LOGGER.error("Another ChudGPT Discord bot instance is already running; exiting duplicate process.")
        return
    # Deployment secrets are loaded only when launching the process. Merely
    # importing this module for tests or tooling never reads the local .env.
    load_dotenv()
    settings = Settings.from_environment()
    state: dict[str, Any] = {"discord_ready": False, "api_status": "unknown", "started_at": int(time.time())}
    soundboard = SoundboardController(settings.soundboard_dir)
    health_app = create_health_app(state, soundboard)
    threading.Thread(
        target=run_health_server,
        args=(health_app, settings.port, settings.soundboard_host),
        daemon=True,
    ).start()

    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    client = discord.Client(intents=intents)
    state["discord_client"] = client
    public_api = ChudGPTClient(settings.api_url, settings.request_timeout)
    translator = GoogleTranslateClient(settings.google_translate_api_key)
    limiter = SlidingWindowLimiter(settings.max_requests_per_minute)
    safe_mentions = discord.AllowedMentions(users=True, roles=False, everyone=False, replied_user=True)
    recent_user_messages: dict[tuple[int, int], deque[str]] = defaultdict(lambda: deque(maxlen=3))
    recent_bot_messages: dict[tuple[int, int], deque[str]] = defaultdict(lambda: deque(maxlen=2))
    language_preferences: dict[tuple[int, int], str] = {}
    server_admin_confirmations: dict[
        tuple[int, int, str], tuple[str, float, int, dict[str, Any] | None]
    ] = {}
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
        state["discord_loop"] = __import__("asyncio").get_running_loop()
        state["bot_user"] = str(client.user)
        state["guild_count"] = len(client.guilds)
        LOGGER.info("Logged in as %s in %d guild(s)", client.user, len(client.guilds))
        await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name=f"{settings.prefix} or mentions"))

    @client.event
    async def on_disconnect() -> None:
        state["discord_ready"] = False

    @client.event
    async def on_guild_join(guild: discord.Guild) -> None:
        """Privately onboard the server owner when ChudGPT is added."""
        owner = guild.owner
        if owner is None:
            try:
                owner = await guild.fetch_member(guild.owner_id)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                owner = None
        if owner is None:
            LOGGER.warning("Could not resolve owner for newly joined guild %s", guild.id)
            return
        try:
            await owner.send(
                f"Thanks for adding **ChudGPT** to **{guild.name}**. "
                f"Use `{settings.prefix} help` for normal commands and uppercase "
                f"`{settings.prefix} SERVER` for restricted server-owner/Administrator tools."
            )
        except discord.Forbidden:
            LOGGER.info("Guild owner %s has DMs disabled; onboarding DM skipped", owner.id)
        except discord.HTTPException as error:
            LOGGER.warning("Could not send onboarding DM for guild %s: %s", guild.id, error)

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
        blacklisted_ids, blacklist_message = load_user_blacklist(settings.blacklist_file)
        if message.author.id in blacklisted_ids:
            await message.reply(
                blacklist_message,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            return
        prompt = clean_prompt(message.content, client.user.id, settings.prefix)
        if not prompt:
            await message.reply(f"Send a message after `{settings.prefix}` or after mentioning me.", mention_author=False)
            return
        if is_memory_clear_request(prompt):
            try:
                await __import__("asyncio").to_thread(public_api.clear, make_session_id(message))
                recent_user_messages.pop(context_key, None)
                recent_bot_messages.pop(context_key, None)
                language_preferences.pop(context_key, None)
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
        if server_admin_action(prompt) is None and not limiter.allow(message.author.id):
            await message.reply("You are sending messages a little too quickly. Try again in about a minute.", mention_author=False)
            return
        try:
            server_admin_reply = await handle_server_admin_command(
                prompt,
                message,
                settings.prefix,
                settings.server_backup_dir,
                server_admin_confirmations,
            )
            if server_admin_reply is not None:
                if server_admin_reply:
                    await message.reply(
                        server_admin_reply,
                        mention_author=False,
                        allowed_mentions=safe_mentions,
                    )
                return
            admin_help_page = requested_admin_help_page(prompt)
            if admin_help_page is not None:
                if message.author.id not in settings.soundboard_admin_user_ids:
                    await message.reply(
                        "Only Astra, the configured ChudGPT owner, can open ADMIN-HELP.",
                        mention_author=False,
                    )
                    return
                view = AdminHelpPaginationView(
                    settings.prefix, admin_help_page, settings.soundboard_admin_user_ids
                )
                view.message = await message.reply(
                    discord_admin_help_page(settings.prefix, admin_help_page),
                    view=view,
                    mention_author=False,
                    allowed_mentions=safe_mentions,
                )
                return
            soundboard_reply = await handle_soundboard_command(
                prompt, message, client, soundboard, settings.soundboard_admin_user_ids, settings.port
            )
            if soundboard_reply is not None:
                await message.reply(soundboard_reply, mention_author=False, allowed_mentions=safe_mentions)
                return
            recent_context = list(recent_user_messages[context_key])
            model_prompt = prompt
            translation_command = parse_translation_command(prompt)
            if translation_command:
                action, value, text_to_translate = translation_command
                if action == "status":
                    current = language_preferences.get(context_key, "auto")
                    reply = f"Translation is available through {translator.provider}. This conversation's language mode is `{current}`."
                elif action == "invalid":
                    reply = f"I don't recognize `{value}` as a configured language. Use `{settings.prefix} languages` for the basic list."
                elif action == "set":
                    if value == "off":
                        language_preferences[context_key] = "off"
                        reply = "Translation is off for your conversation in this channel."
                    else:
                        language_preferences[context_key] = value or "auto"
                        reply = f"Translation mode is now `{value}` for your conversation in this channel."
                else:
                    reply, _ = await __import__("asyncio").to_thread(
                        translator.translate, text_to_translate or "", value or "en"
                    )
                await message.reply(reply, mention_author=False, allowed_mentions=safe_mentions)
                return
            selected_language = language_preferences.get(context_key, "auto")
            response_language: str | None = None
            if translator.enabled and selected_language != "off":
                should_translate_input = selected_language not in {"auto", "en"} or has_non_ascii_letters(prompt)
                if should_translate_input:
                    source = None if selected_language == "auto" else selected_language
                    model_prompt, detected = await __import__("asyncio").to_thread(
                        translator.translate, prompt, "en", source
                    )
                    response_language = selected_language if selected_language != "auto" else detected
            discord_context = discord_identity_context(message, developer_user_id)
            visible_roles = [item.name for item in getattr(message.author, "roles", []) if item.name != "@everyone"]
            visible_server = message.guild.name if message.guild is not None else "a private Discord DM"
            visible_speaker = getattr(message.author, "display_name", None) or message.author.name
            if recent_context:
                discord_context += "; recent same-user messages=" + " | ".join(recent_context[-2:])
            if recent_bot_messages[context_key]:
                discord_context += "; recent bot replies=" + " | ".join(recent_bot_messages[context_key])
            recent_user_messages[context_key].append(prompt)
            whois_reply: str | None = None
            target_id = whois_target_id(prompt)
            if target_id is not None:
                if message.guild is None:
                    whois_reply = "`whois` only works inside a Discord server, not in direct messages."
                else:
                    member = message.guild.get_member(target_id)
                    if member is None:
                        try:
                            member = await message.guild.fetch_member(target_id)
                        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                            member = None
                    whois_reply = (
                        format_whois_member(member)
                        if member is not None
                        else f"I couldn't find a member with ID `{target_id}` in this server."
                    )
            if whois_reply is None:
                whois_reply = malformed_whois_reply(prompt, settings.prefix)
            reply = (
                whois_reply
                or discord_command_reply(
                    prompt, settings.prefix, str(state.get("api_status", "unknown")),
                    visible_speaker, visible_server, visible_roles,
                    getattr(message.channel, "mention", None) or getattr(message.channel, "name", None),
                    message.author.id,
                )
                or discord_quoted_reply(prompt)
                or discord_developer_reply(prompt, developer_user_id)
                or discord_code_reply(model_prompt)
                or discord_social_reply(model_prompt, recent_context)
            )
            if reply is None:
                async with message.channel.typing():
                    reply = await __import__("asyncio").to_thread(
                        public_api.chat, model_prompt, make_session_id(message), discord_context
                    )
            if translator.enabled and response_language and response_language != "en" and not re.search(r"```|<@!?\d+>", reply):
                reply, _ = await __import__("asyncio").to_thread(
                    translator.translate, reply, response_language, "en"
                )
            state["api_status"] = "online"
            await __import__("asyncio").to_thread(
                log_discord_exchange, settings.conversation_log_dir, message, prompt, reply
            )
            recent_bot_messages[context_key].append(reply)
            help_page = requested_help_page(prompt)
            if help_page is not None:
                view = HelpPaginationView(settings.prefix, help_page, message.author.id)
                view.message = await message.reply(
                    discord_help_page(settings.prefix, help_page),
                    view=view,
                    mention_author=False,
                    allowed_mentions=safe_mentions,
                )
                return
            for index, chunk in enumerate(split_discord_message(reply)):
                if index == 0:
                    # Put the real Discord mention in the visible response. It
                    # is constructed from Discord's author object, never from
                    # generated model text or a guessed account ID.
                    await message.reply(
                        place_author_mention(prompt, chunk, message.author.mention),
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
