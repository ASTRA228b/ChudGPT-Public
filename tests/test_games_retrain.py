"""Dataset hygiene and absence of game/unknown-model canned response routing."""
from collections import OrderedDict
from threading import RLock
from types import SimpleNamespace

import pytest

from build_public_games_retrain import rejection_reason, row
from public_api_server import PublicModelService
from public_identity import project_identity_response


@pytest.mark.parametrize('answer,reason', [
    ("I'm sorry, I don't understand.", 'generic-nonanswer'),
    ('<assistant>: training text', 'encoding-or-prompt-leak'),
    ('broken \ufffd text', 'encoding-or-prompt-leak'),
    ('```python\nprint(1)', 'unclosed-code'),
    ('user_id: 123456789', 'private-log-fields'),
    ('I am ChatGPT, an OpenAI model.', 'wrong-assistant-or-personal-experience'),
])
def test_training_rejects_broken_examples(answer,reason):
    assert rejection_reason(row('test','Hello',answer))==reason


@pytest.mark.parametrize('answer', [
    'The console has formed a choir. Start with the first error.',
    'You branch like the tree filed a restraining order.',
    'My imaginary toaster just declared itself mayor. 🦍',
])
def test_humor_and_weirdness_are_kept(answer):
    assert rejection_reason(row('humor','Say something silly',answer)) is None


@pytest.mark.parametrize('prompt', [
    'What is Gorilla Tag?', 'Talk about the Gorilla Tag modding community.',
    'What is BepInEx?', 'What is ChudGPT Galactic Ultra 9000?',
])
def test_games_and_unknown_models_remain_generated(prompt):
    service=object.__new__(PublicModelService)
    service.sessions=OrderedDict()
    service.lock=RLock()
    service.parameters=20999184
    service.model=SimpleNamespace(config=SimpleNamespace(context_length=1024))
    service.system_prompt='You are ChudGPT-Public.'
    service._generate_raw=lambda *args: 'A funny, imperfect generated answer about Galactic Ultra 9000.'
    _,answer=service.chat(prompt,None)
    assert answer=='A funny, imperfect generated answer about Galactic Ultra 9000.'
    assert service.last_assistance_reason is None


def test_unknown_model_name_does_not_get_identity_replacement():
    assert project_identity_response('What is ChudGPT Galactic Ultra 9000?') is None
