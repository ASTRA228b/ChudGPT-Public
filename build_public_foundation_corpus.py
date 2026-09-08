"""Build a coherent-language foundation corpus for the 21M Public model.

TinyStories is used as language-model pretraining data, not as canned runtime
answers.  The fixed seed and row cap keep the experiment reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data/external/tinystories_train_00000.parquet"
DEFAULT_OUTPUT = ROOT / "data/public_foundation_conversations.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stories", type=int, default=120_000)
    args = parser.parse_args()

    table = pq.read_table(args.input, columns=["text"])
    stories = [str(value).strip() for value in table.column("text").to_pylist()]
    stories = [story for story in stories if len(story) >= 80]
    random.Random(20260910).shuffle(stories)
    selected = stories[: args.stories]
    if len(selected) < args.stories:
        raise ValueError(f"Only {len(selected):,} usable stories were available")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for story in selected:
            handle.write(json.dumps({"source": "tinystories", "text": story}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected):,} coherent-language documents to {args.output}")


if __name__ == "__main__":
    main()
