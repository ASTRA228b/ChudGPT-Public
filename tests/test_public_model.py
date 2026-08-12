from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from chudlm.config import dataclass_from_dict, load_yaml
from chudlm.model import ModelConfig, TransformerLM

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT


def test_public_model_has_exact_expected_size() -> None:
    config = dataclass_from_dict(ModelConfig, load_yaml(PUBLIC / "configs" / "model.yaml"))
    model = TransformerLM(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 20_999_184
    assert config.context_length == 1024
    assert config.vocab_size == 8192


def test_public_dataset_is_large_clean_and_reproducible() -> None:
    path = PUBLIC / "data" / "public_conversations.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 20_000
    assert sum(len(record["messages"]) for record in records) == 60_000
    assert all(message["role"] in {"user", "assistant"} and message["content"].strip() for record in records for message in record["messages"])
    assert all(record["source"] == "chudgpt-public-custom" for record in records)
    assistant_text = "\n".join(message["content"] for record in records for message in record["messages"] if message["role"] == "assistant")
    assert "20,999,184 trainable parameters" in assistant_text
    assert "Paris is the capital of France" in assistant_text
    assert "The answer is 12." in assistant_text
    assert "```c#" in assistant_text


def test_public_tokenizer_matches_model() -> None:
    tokenizer = Tokenizer.from_file(str(PUBLIC / "artifacts" / "tokenizer.json"))
    assert tokenizer.get_vocab_size() == 8192
    decoded = tokenizer.decode(tokenizer.encode("Hello, I am ChudGPT and this is readable English.").ids)
    assert "ChudGPT" in decoded and "readable English" in decoded


def test_vercel_api_and_frontend_contract() -> None:
    proxy = (PUBLIC / "api" / "[...path].js").read_text(encoding="utf-8")
    page = (PUBLIC / "public" / "index.html").read_text(encoding="utf-8")
    deployment = json.loads((PUBLIC / "vercel.json").read_text(encoding="utf-8"))
    assert "CHUDGPT_BACKEND_URL" in proxy
    assert "Authorization" in proxy
    assert "/api/chat" in (PUBLIC / "public" / "app.js").read_text(encoding="utf-8")
    assert "ChudGPT-Public" in page
    assert deployment["rewrites"][0]["destination"] == "/public/index.html"
