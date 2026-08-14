from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from chudlm.prompts import DEFAULT_SYSTEM_PROMPT, build_context_token_ids, format_conversation
from public_api_server import PublicModelService
from short_prompt_benchmark import CASES


def test_system_prompt_and_roles_are_stable() -> None:
    formatted = format_conversation(
        [{"role": "user", "content": "hello"}], add_assistant_prompt=True
    )
    assert formatted.startswith(f"<system>: {DEFAULT_SYSTEM_PROMPT}")
    assert "<user>: hello" in formatted
    assert formatted.endswith("<assistant>:")


def test_context_keeps_recent_history_and_system_prompt() -> None:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file("artifacts/tokenizer.json")
    prompt, ids = build_context_token_ids(
        tokenizer,
        [
            {"role": "user", "content": "My mug is coral."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "What color is it?"},
        ],
        1024,
    )
    assert DEFAULT_SYSTEM_PROMPT in prompt
    assert "My mug is coral." in prompt
    assert "What color is it?" in prompt
    assert len(ids) <= 1024


def test_runtime_assistance_is_narrow_and_auditable() -> None:
    source = Path("public_api_server.py").read_text(encoding="utf-8")
    forbidden = (
        "ExampleRetriever", "score_generated_reply", "classify_intent",
        "_calculate_arithmetic", "_reference_answer", "_comparison_answer",
        "_self_answer", "_greeting", "_joke_answer", "_random_code_answer",
        "_short_followup", "_correction_reply", "session_facts",
    )
    for symbol in forbidden:
        assert symbol not in source
    assert "raw_model_generation" in source
    assert "_assist_identity" in source
    assert "stable-public-identity" in source
    assert "stable-family-metadata" in source
    assert "_calculate_arithmetic" not in source
    assert "_random_code_answer" not in source


def test_technical_retry_does_not_invent_content() -> None:
    source = Path("public_api_server.py").read_text(encoding="utf-8")
    assert "Model produced empty output after three generation attempts" in source
    assert 'reply = "..."' not in source
    assert "could not form a useful answer" not in source.lower()
    assert "try asking another way" not in source.lower()


def test_v9_dataset_is_clean_unique_and_large() -> None:
    rows = [json.loads(line) for line in Path("data/public_v9_conversations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 9_000
    fingerprints = {json.dumps(row["messages"], sort_keys=True, ensure_ascii=False) for row in rows}
    assert len(fingerprints) == len(rows)
    text = "\n".join(message["content"] for row in rows for message in row["messages"])
    assert "�" not in text
    assert "Ã" not in text
    assert "one useful way into" not in text.lower()
    assert "the exact joke still depends" not in text.lower()


def test_v9_exact_assistant_repetition_is_capped() -> None:
    rows = [json.loads(line) for line in Path("data/public_v9_conversations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    answers = [message["content"] for row in rows for message in row["messages"] if message["role"] == "assistant"]
    assert max(Counter(answers).values()) <= 4


def test_heldout_cases_are_not_in_training_data() -> None:
    rows = [json.loads(line) for line in Path("data/public_v9_conversations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    training_prompts = {message["content"].strip().lower() for row in rows for message in row["messages"] if message["role"] == "user"}
    heldout_prompts = {turn.strip().lower() for case in CASES for turn in case.turns}
    assert not training_prompts.intersection(heldout_prompts)


def test_public_model_service_has_raw_generation_method() -> None:
    assert hasattr(PublicModelService, "_generate_raw")


def test_identity_assistance_does_not_route_normal_topics() -> None:
    normal = ("hello", "67", "67 + 8", "write Python", "tell me a joke", "tung tung sahur")
    for prompt in normal:
        assert PublicModelService._identity_subject(prompt) is None
    assert PublicModelService._identity_subject("What are you?") == "public"
    assert PublicModelService._identity_subject("What is ChudGPT Pro?") == "pro"
    assert PublicModelService._identity_subject("What other ChudGPTs exist?") == "family"


def test_v10_dataset_is_balanced_unique_and_large() -> None:
    rows = [json.loads(line) for line in Path("data/public_v10_conversations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 12_000
    fingerprints = {json.dumps(row["messages"], sort_keys=True, ensure_ascii=False) for row in rows}
    assert len(fingerprints) == len(rows)
    user_text = [" ".join(message["content"] for message in row["messages"] if message["role"] == "user").lower() for row in rows]
    math_rows = sum(bool(__import__("re").search(r"calculate|arithmetic|factorial|percent|\d\s*[+*/x-]\s*\d", text)) for text in user_text)
    assert math_rows <= 1_700
    assert sum("chudgpt" in text or "who are you" in text or "what are you" in text for text in user_text) >= 90
