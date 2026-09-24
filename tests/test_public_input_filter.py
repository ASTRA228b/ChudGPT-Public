import pytest

from public_input_filter import filter_racial_slurs, SLUR_ONLY_REPLY
from test_awareness_recovery import service


@pytest.mark.parametrize("message", ["nigger", "NIGGA!", "...kike...", "wetbacks ragheads", "nigger 😡"])
def test_slur_only_is_not_generated_or_saved(service, message):
    session, reply = service.chat(message, None)
    assert reply == SLUR_ONLY_REPLY
    assert service.last_assistance_reason == "racial_slur_filter"
    assert service.last_neural_profile is None
    assert session not in service.sessions


def test_mixed_request_uses_existing_math_handler(service):
    session, reply = service.chat("What is 2 + 2? nigger", None)
    assert "4" in reply
    assert service.last_assistance_reason == "exact_math"
    assert "nigger" not in str(service.sessions[session])


def test_neural_request_and_discord_context_are_cleaned(service):
    captured = []
    service._generate_raw = lambda *args: captured.append(args) or "NEURAL"
    session, reply = service.chat("Explain gravity nigger", None, context_mode="discord", discord_context="someone said kike")
    assert reply == "NEURAL"
    assert "nigger" not in str(captured) and "kike" not in str(captured)
    assert "Explain gravity" in str(captured)
    assert service.last_assistance_reason is None
    service.sessions[session].append({"role": "user", "content": "old wetback text"})
    service.chat("Tell me more", session)
    assert "wetback" not in str(captured)


@pytest.mark.parametrize("message", [
    "Black people", "I'm gay", "I'm trans", "Pakistani food", "spicy rice",
    "sniggering", "a chink in the armor", "こんにちは", "🏳️‍🌈",
])
def test_innocent_text_is_unchanged(message):
    assert filter_racial_slurs(message) == (message, False)


def test_blocked_turn_preserves_existing_session(service):
    session, _ = service.chat("What is 2+2?", None)
    before = list(service.sessions[session])
    assert service.chat("nigger", session)[1] == SLUR_ONLY_REPLY
    assert service.sessions[session] == before
