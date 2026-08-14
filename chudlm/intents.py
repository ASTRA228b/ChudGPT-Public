"""Conservative shared intent classification for ChudGPT Public."""

from __future__ import annotations

import re
from dataclasses import dataclass

NEGATIVE_PATTERNS = (
    r"\bno\s+math\b", r"\bnot\s+math\b", r"\bdon['’]?t\s+(?:do|give|use|show)\s+(?:me\s+)?math\b",
    r"\bno\s+code\b", r"\bdon['’]?t\s+(?:give|write|show|send|use)\s+(?:me\s+)?code\b",
    r"\bnot\s+that\b", r"\bi\s+didn['’]?t\s+ask\s+for\s+that\b",
    r"\bstop\s+(?:explaining|calculating|coding)\b",
)


@dataclass(frozen=True)
class IntentResult:
    name: str
    confidence: float
    is_correction: bool = False


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9+#.%*/×÷'’+-]+", text.lower()))


def has_negative_override(text: str) -> bool:
    return any(re.search(pattern, text.lower()) for pattern in NEGATIVE_PATTERNS)


def has_strong_math_intent(text: str) -> bool:
    lowered = normalize(text)
    if has_negative_override(lowered):
        return False
    expression = bool(re.search(r"(?<!\w)-?\d+(?:\.\d+)?\s*(?:\+|\*|/|×|÷|-(?=\s*-?\d))\s*-?\d+(?:\.\d+)?(?!\w)", lowered))
    directive = bool(re.search(r"\b(?:calculate|compute|solve|evaluate|work out|what is|how much is|find)\b", lowered))
    word_operation = bool(re.search(r"\b\d+(?:\.\d+)?\s+(?:plus|minus|times|multiplied by|divided by)\s+\d+(?:\.\d+)?\b", lowered))
    percent = bool(re.search(r"\b(?:percent|percentage|discount|interest rate)\b|%\s+of", lowered) and re.search(r"\d", lowered))
    conversion = bool(re.search(r"\b(?:convert|conversion|how many)\b", lowered) and re.search(r"\d", lowered) and re.search(r"\b(?:miles?|kilometers?|km|kg|pounds?|hours?|minutes?|celsius|fahrenheit)\b", lowered))
    word_problem = bool(re.search(r"\d", lowered) and re.search(r"\b(?:total|each|remaining|left|distance|speed|mph|cost|change|average|area|perimeter)\b", lowered) and text.rstrip().endswith("?"))
    return word_operation or percent or conversion or word_problem or (expression and (directive or len(lowered.split()) <= 7))


def classify_intent(text: str) -> IntentResult:
    lowered = normalize(text)
    if has_negative_override(lowered):
        return IntentResult("correction", 0.99, True)
    if has_strong_math_intent(text):
        return IntentResult("math", 0.98)
    tokens = set(lowered.split())
    if tokens & {"code", "python", "c#", "csharp", "javascript", "unity", "sql", "debug", "program", "script", "rust"}:
        return IntentResult("code", 0.92)
    if tokens & {"meme", "rickroll", "brainrot", "amogus", "wojak", "doge", "pepe", "rizz", "aura", "skibidi", "gigachad", "chad"} or lowered == "67" or re.search(r"\b(?:this is peak|what does (?:cooked|peak|real) mean|bro is not|average discord mod|i['’]?m cooked|virgin (?:vs|versus) chad|let (?:him|her|them) cook)\b", text, re.I):
        return IntentResult("meme", 0.90)
    if any(phrase in lowered for phrase in ("who are you", "what are you", "your name", "chudgpt", "what can you do")):
        return IntentResult("identity", 0.95)
    if len(lowered.split()) <= 3:
        return IntentResult("conversation", 0.45)
    if text.rstrip().endswith("?"):
        return IntentResult("question", 0.65)
    return IntentResult("conversation", 0.60)
