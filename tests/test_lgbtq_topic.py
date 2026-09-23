import pytest
from public_lgbtq_topic import lgbtq_conversation
from test_awareness_recovery import service


@pytest.mark.parametrize('prompt', ["I'm trans", 'im gay', 'Im a femboy', 'are you not gay', 'my friend came out as trans'])
def test_conversation_topic(prompt):
    assert lgbtq_conversation(prompt)


@pytest.mark.parametrize('prompt', ['Help me with math', 'What is Gorilla Tag?', 'Write a gay pride poem', 'debug transform code', 'translate gay to French'])
def test_other_requests_keep_main_model(prompt):
    assert not lgbtq_conversation(prompt)


def test_neural_specialist_only_handles_its_topic(service):
    from types import SimpleNamespace
    service.lgbtq_service = SimpleNamespace(_generate_raw=lambda *args: 'A generated topical reply.')
    session, reply = service.chat("I'm trans", None)
    assert reply == 'A generated topical reply.'
    assert service.last_assistance_reason is None
    assert service.chat('What is Gorilla Tag?', session)[1] == 'NEURAL'


def test_followup_needs_explicit_recent_topic():
    assert lgbtq_conversation('what about you?', [{'role':'user','content':'im gay'}])
    assert not lgbtq_conversation('what about you?', [{'role':'user','content':'I like Minecraft'}])
