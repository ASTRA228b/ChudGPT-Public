import json
from pathlib import Path

from benchmark_balanced_public import CASES, CONFIRMATION_CASES
from chudlm.sft_data import split_records


def test_related_conversations_do_not_cross_validation_boundary():
    records = []
    for name in ("Oak", "Pine", "Birch", "Elm", "Ash"):
        short = [{"role": "user", "content": f"My project is {name}."},
                 {"role": "assistant", "content": f"Okay, {name}."}]
        records.extend([short, short + [{"role": "user", "content": "What is it called?"},
                                       {"role": "assistant", "content": name}]])
    train, validation = split_records(records, .3, 123)
    assert {r[0]["content"] for r in train}.isdisjoint({r[0]["content"] for r in validation})
    assert len(train) + len(validation) == len(records)


def test_benchmark_prompts_are_not_training_examples():
    rows = [json.loads(line) for path in ("data/public_balanced_reviewed.jsonl", "data/public_balanced_binding.jsonl")
            for line in Path(path).read_text(encoding="utf-8").splitlines()]
    training_prompts = {m["content"].strip().casefold() for row in rows for m in row["messages"] if m["role"] == "user"}
    held_out = {prompt.strip().casefold() for _, prompts, _ in CASES + CONFIRMATION_CASES for prompt in prompts}
    assert not training_prompts.intersection(held_out)
