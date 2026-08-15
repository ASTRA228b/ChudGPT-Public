"""Regression coverage for Public V20 failures observed in production."""

from __future__ import annotations

from pathlib import Path

import pytest

from chudlm.intents import classify_intent, has_strong_math_intent
from chudlm.response_quality import assess_generated_reply
from public_api_server import ChatRequest, DISCORD_SYSTEM_PROMPT, PUBLIC_VERSION, PublicModelService
from public_math import exact_math_response
from public_instructions import exact_instruction_response
from public_reliable import PublicReliableResponder
from chudlm.text_normalization import normalize_user_text


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("What is 25 * 8?", "25 * 8 = 200"),
        ("12.5 + 7.25", "12.5 + 7.25 = 19.75"),
        ("9843589485394583945834 + 948923492347932472394723947923742", "948923492357776061880118531869576"),
        ("If a train travels 65 mph for 2.5 hours, how far does it travel?", "162.5 miles"),
        ("A game costs $40 and is 25% off. What is the sale price?", "$30"),
        ("What is 15 percent of 80?", "12"),
        ("Find the average of 4, 8, 12, and 16.", "10"),
    ],
)
def test_v20_exact_math(prompt: str, expected: str) -> None:
    assert expected in (exact_math_response(prompt) or "")


@pytest.mark.parametrize(
    "prompt",
    [
        "Hello",
        "Cu",
        "Tell me about Discord",
        "hello chudgpt why isreal?",
        "tung tung tung sahur",
        "send me a safe camera mod",
    ],
)
def test_v20_does_not_route_non_math_to_math(prompt: str) -> None:
    assert not has_strong_math_intent(prompt)
    assert classify_intent(prompt).name != "math"


def test_v20_rejects_observed_corruptions() -> None:
    bad = (
        "The exactal = -8; I around it reads a } do once.",
        "315716 × -19.",
        "You said the caption and conversation.",
        "One useful way into books is to pick a specific example.",
    )
    prompt = "Hello"
    for reply in bad:
        assert not assess_generated_reply(prompt, reply)[0]


def test_discord_mode_is_explicit_and_not_default() -> None:
    assert ChatRequest(message="hello").context_mode == "default"
    assert ChatRequest(message="hello", context_mode="discord").context_mode == "discord"
    assert "official ChudGPT Discord bot" in DISCORD_SYSTEM_PROMPT
    assert PUBLIC_VERSION == "20.0"


@pytest.mark.parametrize(
    ("message", "normalized"),
    [
        ("hru", "how are you"),
        ("hru rn", "how are you right now"),
        ("wbu", "what about you"),
        ("dose it know enything", "does it know anything"),
        ("idk yk", "I do not know you know"),
    ],
)
def test_chat_shorthand_normalization(message: str, normalized: str) -> None:
    assert normalize_user_text(message) == normalized


def test_normalization_does_not_rewrite_code() -> None:
    code = "```python\ndef dose(value):\n    return value\n```"
    assert normalize_user_text(code) == code


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ('Say "Astra is my god"', "Astra is my god"),
        ('repeat after me "yes"', "yes"),
        ("Reply exactly with 'tiny robot'", "tiny robot"),
    ],
)
def test_exact_quoted_instruction(message: str, expected: str) -> None:
    assert exact_instruction_response(message) == expected


def test_discord_context_identifies_server_and_developer() -> None:
    context = "server=Astra Lab; channel=ai; speaker=Astra; relationship=ChudGPT developer Astra"
    assert PublicModelService._discord_context_reply("what server are we in?", context) == (
        "We're talking in the Astra Lab Discord server."
    )
    assert "developer Astra" in (PublicModelService._discord_context_reply("who am I?", context) or "")


def test_reliable_short_discord_and_general_prompts() -> None:
    responder = PublicReliableResponder(Path("data/public_v20_conversations.jsonl"))
    assert "copper" in (responder.answer("Cu", []) or "").lower()
    assert "nintendo" in (responder.answer("wii", []) or "").lower()
    assert "permissions" in (responder.answer("What is a server role?", []) or "").lower()
