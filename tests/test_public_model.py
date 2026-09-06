from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from chudlm.prompts import DEFAULT_SYSTEM_PROMPT, build_context_token_ids, format_conversation
from public_api_server import PublicModelService, selected_checkpoint
from public_meme_facts import find_meme_fact
from public_math import exact_integer_arithmetic
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


def test_runtime_is_neural_only_and_auditable() -> None:
    source = Path("public_api_server.py").read_text(encoding="utf-8")
    forbidden = (
            "ExampleRetriever", "classify_intent",
        "_calculate_arithmetic", "_reference_answer", "_comparison_answer",
        "_self_answer", "_greeting", "_joke_answer", "_random_code_answer",
        "_short_followup", "_correction_reply", "session_facts",
    )
    for symbol in forbidden:
        assert symbol not in source
    assert "raw_model_generation" in source
    assert "_assist_identity" not in source
    assert "PublicReliableResponder" not in source
    assert "exact_math_response" not in source
    assert "exact_instruction_response" not in source
    assert '"fallbacks": False' in source
    assert "_calculate_arithmetic" not in source
    assert "_random_code_answer" not in source


def test_technical_retry_never_uses_a_canned_answer() -> None:
    source = Path("public_api_server.py").read_text(encoding="utf-8")
    assert "Model repeatedly generated only rejected uncertainty templates" not in source
    assert "never_expose" in source
    assert 'reply = "..."' not in source
    assert "I couldn't form a relevant answer" not in source
    assert "did not produce a usable neural reply" in source
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


def test_serving_config_selects_v20_and_keeps_v8_archived() -> None:
    config = json.loads(Path("serving_config.json").read_text(encoding="utf-8"))
    assert selected_checkpoint() == "checkpoints/public_v20_quality/best.pt"
    assert config["archived_checkpoints"]["v8"] == "checkpoints/public_v8/best.pt"
    assert config["archived_checkpoints"]["v10_balanced"] == "checkpoints/public_v10_balanced/best.pt"
    assert config["archived_checkpoints"]["v18"] == "checkpoints/public_v18_sft/best.pt"
    assert config["archived_checkpoints"]["v20"] == "checkpoints/public_v20_final/best.pt"


def test_reviewed_meme_help_is_narrow_and_unknown_text_stays_neural() -> None:
    assert "absurdist" in (find_meme_fact("Tung tung tung tung tung sahur") or "")
    assert "number" in (find_meme_fact("what does the 67 meme mean?") or "")
    assert find_meme_fact("florble snax 992") is None
    assert find_meme_fact("What is ChudGPT?") is None
    assert find_meme_fact("What is ChudGPT-Public?") is None
    assert "insult" in (find_meme_fact("What does chud mean?") or "")


def test_candidate_ranking_rejects_cross_topic_code_and_math() -> None:
    relevant = "Hey! Good to hear from you. What is on your mind?"
    code_leak = "```python\nprint(2 + 2)\n```"
    assert PublicModelService._candidate_score("Hello mate", relevant) > PublicModelService._candidate_score("Hello mate", code_leak)

    casual = "That sounds unusual; tell me what you mean by it."
    math_leak = "Distance = 75 * 1.5 = 112.5 miles."
    assert PublicModelService._candidate_score("tung tung sahur", casual) > PublicModelService._candidate_score("tung tung sahur", math_leak)


def test_candidate_ranking_rewards_requested_output_type() -> None:
    code = "```csharp\npublic class Player {}\n```"
    chatter = "That is an interesting question about music."
    assert PublicModelService._candidate_score("Write C# code", code) > PublicModelService._candidate_score("Write C# code", chatter)

    result = "The answer is 43."
    unrelated = "Saturn is known for its rings."
    assert PublicModelService._candidate_score("What is 17 plus 26?", result) > PublicModelService._candidate_score("What is 17 plus 26?", unrelated)


def test_exact_integer_arithmetic_handles_large_values() -> None:
    assert exact_integer_arithmetic(
        "9843589485394583945834 + 948923492347932472394723947923742"
    ) == "9843589485394583945834 + 948923492347932472394723947923742 = 948923492357776061880118531869576"
    assert exact_integer_arithmetic("Compute 999999999999999999999 - 1") == (
        "999999999999999999999 - 1 = 999999999999999999998"
    )
    assert exact_integer_arithmetic("12345678901234567890 times 9876543210") == (
        "12345678901234567890 * 9876543210 = 121932631124828532111263526900"
    )


def test_exact_integer_division_never_uses_float_precision() -> None:
    assert exact_integer_arithmetic("100000000000000000000 / 4") == (
        "100000000000000000000 / 4 = 25000000000000000000"
    )
    assert exact_integer_arithmetic("7 divided by 2") == "7 / 2 = 3.5"
    assert exact_integer_arithmetic("10 / 3") == "10 / 3 = 10/3"
    assert exact_integer_arithmetic("5 / 0") == "Division by zero is undefined."


def test_exact_math_gate_does_not_capture_non_math_prompts() -> None:
    non_math = (
        "67", "hello", "What does 67 mean?", "Python raises ZeroDivisionError for 5/0—why?",
        "The 2024-2025 season was wild", "write code that adds two numbers",
    )
    assert all(exact_integer_arithmetic(prompt) is None for prompt in non_math)


def test_v10_dataset_is_balanced_unique_and_large() -> None:
    rows = [json.loads(line) for line in Path("data/public_v10_conversations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 12_000
    fingerprints = {json.dumps(row["messages"], sort_keys=True, ensure_ascii=False) for row in rows}
    assert len(fingerprints) == len(rows)
    user_text = [" ".join(message["content"] for message in row["messages"] if message["role"] == "user").lower() for row in rows]
    math_rows = sum(bool(__import__("re").search(r"calculate|arithmetic|factorial|percent|\d\s*[+*/x-]\s*\d", text)) for text in user_text)
    assert math_rows <= 1_700
    assert sum("chudgpt" in text or "who are you" in text or "what are you" in text for text in user_text) >= 90
