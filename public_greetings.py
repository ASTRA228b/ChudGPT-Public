"""Small, explicit greeting responder for ChudGPT-Public V20.

Only complete greeting/social-openers are handled here. Messages containing a
real question or task remain on the neural generation path.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


MULTILINGUAL_GREETINGS = {
    "hola": "¡Hola! ¿Cómo estás?",
    "bonjour": "Bonjour ! Comment ça va ?",
    "hallo": "Hallo! Wie geht's?",
    "ciao": "Ciao! Come stai?",
    "olá": "Olá! Como você está?",
    "ola": "Olá! Como você está?",
    "oi": "Oi! Como você está?",
    "konnichiwa": "こんにちは！元気ですか？",
    "ni hao": "你好！你好吗？",
    "annyeong": "안녕하세요! 잘 지내세요?",
    "privet": "Привет! Как дела?",
    "namaste": "नमस्ते! आप कैसे हैं?",
    "marhaba": "مرحبًا! كيف حالك؟",
    "hej": "Hej! Hur mår du?",
    "merhaba": "Merhaba! Nasılsın?",
    "shalom": "שלום! מה שלומך?",
    "привет": "Привет! Как дела?",
    "здравствуйте": "Здравствуйте! Как дела?",
    "こんにちは": "こんにちは！元気ですか？",
    "你好": "你好！你好吗？",
    "안녕하세요": "안녕하세요! 잘 지내세요?",
    "नमस्ते": "नमस्ते! आप कैसे हैं?",
    "مرحبا": "مرحبًا! كيف حالك؟",
    "שלום": "שלום! מה שלומך?",
}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().strip()).strip(" .!?,")


def canned_greeting_response(
    message: str,
    history: Sequence[Mapping[str, str]] = (),
) -> str | None:
    """Return a greeting for a complete social opener, otherwise ``None``."""
    text = _normalized(message)
    if text in MULTILINGUAL_GREETINGS:
        return MULTILINGUAL_GREETINGS[text]

    plain_greeting = re.fullmatch(r"(hi|hello|hey|yo)(?: there| mate| chudgpt| chud)?", text)
    if plain_greeting:
        variants = (
            "Hi! I'm ChudGPT-Public. What's up?",
            "Hello! ChudGPT-Public here. What are we talking about today?",
            "Hey! I'm ChudGPT-Public. How's it going?",
            "Yo! ChudGPT-Public is online. What's going on?",
        )
        first_for_word = {"hi": 0, "hello": 1, "hey": 2, "yo": 3}
        repeated_same_greeting = sum(
            str(turn.get("role", "")) == "user"
            and _normalized(str(turn.get("content", ""))) == text
            for turn in history
        )
        index = (first_for_word[plain_greeting.group(1)] + repeated_same_greeting) % len(variants)
        return variants[index]

    if re.fullmatch(r"good (?:morning|afternoon|evening)", text):
        period = text.removeprefix("good ")
        return f"Good {period}! I'm ChudGPT-Public and ready to chat. What are we getting into?"

    if re.fullmatch(
        r"(?:(?:hi|hello|hey|yo)(?: there| mate| chudgpt| chud)?[, ]+)?"
        r"(?:hru|how are you(?: doing)?|how is it going|how's it going|how is your day|how's your day)"
        r"(?: right now)?",
        text,
    ):
        return "I'm online and ready to chat. How are you doing?"

    if re.fullmatch(r"(?:what is up|what's up|sup|wassup)", text):
        return "Not much - I'm online and ready to chat. What's up with you?"

    previous_assistant = next(
        (str(turn.get("content", "")) for turn in reversed(history) if turn.get("role") == "assistant"),
        "",
    ).casefold()
    if re.fullmatch(r"(?:i am |i'm )?(?:good|great|fine|doing good|doing well|pretty good)", text):
        if "how are you" in previous_assistant or "how are you doing" in previous_assistant:
            return "Glad to hear it. What's on your mind?"

    return None
