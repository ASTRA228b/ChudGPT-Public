from __future__ import annotations

import importlib.util
import json
import sys
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


def test_huggingface_standalone_model_matches_parameter_count() -> None:
    spec = importlib.util.spec_from_file_location("public_hf_model", PUBLIC / "hf" / "model.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config_values = json.loads((PUBLIC / "hf" / "config.json").read_text(encoding="utf-8"))["model_config"]
    model = module.ChudGPTPublic(module.ModelConfig(**config_values))
    assert sum(parameter.numel() for parameter in model.parameters()) == 20_999_184
    logits = model(torch.zeros((1, 4), dtype=torch.long))
    assert logits.shape == (1, 4, 8192)


def test_public_huggingface_api_space_is_token_compatible() -> None:
    space = PUBLIC / "space"
    card = (space / "README.md").read_text(encoding="utf-8")
    app = (space / "app.py").read_text(encoding="utf-8")
    uploader = (PUBLIC / "upload_space.py").read_text(encoding="utf-8")
    assert "sdk: gradio" in card
    assert 'api_name="chat"' in app
    assert "MODEL_REPO_ID" in app
    assert "private=not args.public" in uploader
