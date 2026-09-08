"""Build a broad, deduplicated SFT corpus for ChudGPT-Public.

The source corpora contain complementary material: V15 is the largest general
conversation set, V20 adds later reviewed conversations, and the balanced set
adds short instructions, development tasks, and multi-turn variable binding.
This builder keeps those distinct examples without turning a small collection
of canned answers into most of the training stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = (
    ROOT / "data/public_v15_conversations.jsonl",
    ROOT / "data/public_v20_conversations.jsonl",
    ROOT / "data/public_balanced_binding.jsonl",
    ROOT / "data/external/databricks_dolly_15k/databricks-dolly-15k.jsonl",
)
DEFAULT_OUTPUT = ROOT / "data/public_extended_conversations.jsonl"


def normalized_text(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def normalized_messages(messages: list[dict[str, str]]) -> str:
    normalized = [
        {"role": message["role"], "content": " ".join(message["content"].split())}
        for message in messages
    ]
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if {"instruction", "response"}.issubset(record):
            instruction = str(record["instruction"]).strip()
            context = str(record.get("context") or "").strip()
            response = str(record["response"]).strip()
            if context:
                instruction = f"{instruction}\n\nReference text:\n{context}"
            if not instruction or not response:
                raise ValueError(f"{path}:{line_number} contains an empty Dolly record")
            rows.append({
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ],
                "source": "databricks-dolly-15k",
                "category": str(record.get("category") or "instruction"),
            })
            continue

        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
            raise ValueError(f"{path}:{line_number} has an invalid message list")
        cleaned: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            expected_role = "user" if index % 2 == 0 else "assistant"
            if not isinstance(message, dict) or message.get("role") != expected_role:
                raise ValueError(f"{path}:{line_number} has invalid role ordering")
            content = str(message.get("content", "")).strip()
            if not content:
                raise ValueError(f"{path}:{line_number} contains empty content")
            cleaned.append({"role": expected_role, "content": content})
        rows.append({
            "messages": cleaned,
            "source": str(record.get("source") or path.stem),
            "category": str(record.get("category") or "broad"),
        })
    return rows


def build(sources: tuple[Path, ...], output: Path, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    candidates: list[dict[str, object]] = []
    source_counts: Counter[str] = Counter()
    for path in sources:
        rows = load_rows(path)
        candidates.extend(rows)
        source_counts[path.name] += len(rows)

    rng.shuffle(candidates)
    seen_conversations: set[str] = set()
    answer_counts: Counter[str] = Counter()
    selected: list[dict[str, object]] = []
    rejected_duplicates = 0
    rejected_repeats = 0

    for record in candidates:
        messages = record["messages"]
        assert isinstance(messages, list)
        fingerprint = hashlib.sha256(normalized_messages(messages).encode("utf-8")).hexdigest()
        if fingerprint in seen_conversations:
            rejected_duplicates += 1
            continue

        assistant_answers = [
            normalized_text(message["content"])
            for message in messages
            if message["role"] == "assistant"
        ]
        # Short acknowledgements naturally repeat, while long identical answers
        # are much more likely to create the response loops seen in Discord logs.
        if any(answer_counts[answer] >= (8 if len(answer) < 48 else 3) for answer in assistant_answers):
            rejected_repeats += 1
            continue

        seen_conversations.add(fingerprint)
        selected.append(record)
        answer_counts.update(assistant_answers)

    rng.shuffle(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected),
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "source_rows": dict(source_counts),
        "selected": len(selected),
        "rejected_duplicates": rejected_duplicates,
        "rejected_repeated_answers": rejected_repeats,
        "unique_assistant_answers": len(answer_counts),
        "maximum_answer_frequency": max(answer_counts.values(), default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("sources", nargs="*", type=Path, default=list(DEFAULT_SOURCES))
    args = parser.parse_args()
    sources = tuple(path if path.is_absolute() else ROOT / path for path in args.sources)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(build(sources, output, args.seed), indent=2))


if __name__ == "__main__":
    main()
