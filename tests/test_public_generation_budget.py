from collections import OrderedDict
from threading import RLock
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer

import public_api_server as api


@pytest.mark.parametrize("prompt", [
    "Generate a simple for loop in Java.",
    "Explain how a refrigerator works.",
    "Compare TCP and UDP.",
    "Write a Python function reversing a string.",
    "Why does ice float?",
])
def test_short_tasks_keep_requested_generation_budget(monkeypatch, prompt):
    service = object.__new__(api.PublicModelService)
    service.tokenizer = Tokenizer.from_file("artifacts/tokenizer.json")
    service.model = SimpleNamespace(config=SimpleNamespace(context_length=2048))
    service.device = torch.device("cpu")
    service.eos_id = service.tokenizer.token_to_id("<eos>")
    service.shorten_casual_generation = True
    budgets = []

    def sample(model, tokens, **kwargs):
        budgets.append(kwargs["max_new_tokens"])
        result = service.tokenizer.encode("This is a generated response.").ids
        return torch.cat((tokens, torch.tensor([result])), dim=1)

    monkeypatch.setattr(api, "generate", sample)
    monkeypatch.setattr(api, "assess_generated_reply", lambda *args: (True, []))
    service._generate_raw([{"role": "user", "content": prompt}], 200, .6, "You are ChudGPT-Public.")
    assert budgets and all(budget == 200 for budget in budgets)


def test_explicit_topic_change_does_not_reuse_stale_exchanges(monkeypatch):
    service = object.__new__(api.PublicModelService)
    service.sessions = OrderedDict()
    service.lock = RLock()
    service.system_prompt = "You are ChudGPT-Public."
    service.parameters = 20999184
    service.model = SimpleNamespace(config=SimpleNamespace(context_length=2048))
    service.sessions["test"] = [
        message for i in range(5) for message in (
            {"role": "user", "content": f"Question {i}"},
            {"role": "assistant", "content": f"Answer {i}"},
        )
    ]
    captured = []
    monkeypatch.setattr(service, "_generate_raw", lambda history, *args: captured.extend(history) or "A generated answer.")
    service.chat("Describe a quiet forest.", "test")
    assert [message["role"] for message in captured] == ["user"]
    assert captured[0]["content"] == "Describe a quiet forest."


def test_generation_history_keeps_explicit_followups_and_related_topics():
    history = [
        {"role": "user", "content": "My robot is named Cedar."},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "What is its name?"},
    ]
    assert api.select_generation_history(history) == history

    related = history[:-1] + [{"role": "user", "content": "Tell me more about the robot."}]
    assert api.select_generation_history(related) == related


def test_generation_history_isolates_unrelated_name_topic():
    history = [
        {"role": "user", "content": "What is my name?"},
        {"role": "assistant", "content": "Its name is Tenda."},
        {"role": "user", "content": "Explain recursion."},
    ]
    assert api.select_generation_history(history) == [history[-1]]


def test_repetition_ranking_preserves_requested_quotes():
    previous = ["A CPU has a few powerful cores for complicated decisions."]
    repeated = "A CPU has a few powerful cores for complicated decisions today."
    fresh = "What part of your game are you working on?"
    penalty = api.PublicModelService._conversation_repetition_penalty
    assert penalty("Tell me more", repeated, previous) > penalty("Tell me more", fresh, previous)
    assert penalty("Repeat that verbatim", repeated, previous) == 0
