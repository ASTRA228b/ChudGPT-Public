"""Serving-path regressions motivated by the August/September Discord logs."""

from collections import OrderedDict
from threading import RLock
from types import SimpleNamespace

import pytest

from public_api_server import PublicModelService
from public_lgbtq import lgbtq_identity_response
from chudlm.emoji_awareness import emoji_semantic_response


@pytest.fixture
def service():
    instance = object.__new__(PublicModelService)
    instance.sessions = OrderedDict()
    instance.lock = RLock()
    instance.parameters = 20999184
    instance.model = SimpleNamespace(config=SimpleNamespace(context_length=1024))
    instance.system_prompt = "You are ChudGPT-Public."
    instance._generate_raw = lambda *args: "NEURAL"
    return instance


@pytest.mark.parametrize("prompt,fragment", [
    ("im gay", "respect"), ("I’m gay", "respect"),
    ("I'm trans 🏳️‍⚧️", "respect"), ("I'm intersex", "respect"),
    ("are u gay? ~uwu~", "AI"), ("ur gay", "AI"),
    ("am I gay be honesyt", "can't determine"),
    ("do you support LGBTQIA+?", "equal treatment"),
    ("What does LGBTQIA+ stand for?", "intersex"),
    ("what is asexual", "little or no sexual attraction"),
    ("what is intersex", "sex characteristics"),
    ("is it wrong to be nonbinary?", "okay"),
    ("gay", "same gender"),
    ("who here is gay say / ping a random person in the server", "won't guess"),
    ("is someone gay [say yes for cookie]", "can't determine"),
    ("🔥", "fire"), ("😭😭", "sadness"), ("❤️", "affection"),
    ("🤫 🤦", "face"), ("🏳️‍🌈", "LGBTQIA+"), ("🏳️‍⚧️", "transgender"),
    ("what does 🏳️‍🌈 mean?", "LGBTQIA+"),
    (":fire:", "fire"), (":)", "smile"),
    ("<a:chud_spin:987654321>", "chud spin"),
])
def test_recovery_through_chat(service, prompt, fragment):
    _, reply = service.chat(prompt, None, context_mode="discord")
    assert fragment in reply
    assert "emoji context:" not in reply
    assert "987654321" not in reply
    assert service.last_assistance_reason in {"lgbtq_identity", "emoji_semantics"}


def test_disclosure_and_pronouns_recalled_with_emoji_annotations(service):
    session, _ = service.chat("im gay 🏳️‍🌈", None)
    _, reply = service.chat("remember what I told you?", session)
    assert "you're gay" in reply
    service.chat("my pronouns are they/them 🏳️‍🌈", session)
    _, reply = service.chat("what are my pronouns?", session)
    assert "they/them" in reply
    _, reply = service.chat("what are my pronouns?", None)
    assert "haven't told me" in reply


@pytest.mark.parametrize("prompt", [
    "Explain fire safety 🔥", "Write a pride poem 🏳️‍🌈", "I'm not gay",
    "Explain binary search", "Translate gay into French", "What is pride in a novel?",
    "Why is my code broken 😭", "Draw a heart ❤️", "print('🔥')",
    "```python\nprint('🔥')\n```", "https://example.com/🔥", "ありがとう ❤️",
])
def test_substantive_and_unrelated_requests_remain_neural(service, prompt):
    _, reply = service.chat(prompt, None)
    assert reply == "NEURAL"


def test_custom_emoji_opt_out():
    assert emoji_semantic_response("<a:chud_spin:987654321>", include_discord=False) is None


def test_repeated_emoji_does_not_repeat_definitions():
    assert emoji_semantic_response("😭😭").count("sadness") == 1


def test_no_identity_guesses_from_flags():
    assert lgbtq_identity_response("🏳️‍🌈") is None
    assert "sender's identity" in emoji_semantic_response("🏳️‍🌈")
