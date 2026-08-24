"""Measure the local cost and tokenizer impact of emoji annotations."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

from tokenizers import Tokenizer

from chudlm.emoji_awareness import EmojiDatabase, add_emoji_context

ROOT = Path(__file__).resolve().parent
MESSAGES = [
    "hello", "bro what 😭💀", "my dog died 😭", "CHUDGPT COOKED 🔥🔥🔥",
    "👍🏽", "👨‍💻", "👨‍👩‍👧‍👦", "🇧🇷 futebol", ":sob: no way",
    "<a:chud_spin:987654321>", "https://example.com/a:b", "```python\nprint(':D')\n```",
]


def main() -> None:
    tracemalloc.start()
    started = time.perf_counter()
    database = EmojiDatabase()
    startup_ms = (time.perf_counter() - started) * 1_000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    repeats = 1_000
    started = time.perf_counter()
    for _ in range(repeats):
        for message in MESSAGES:
            add_emoji_context(message)
    elapsed = time.perf_counter() - started

    tokenizer = Tokenizer.from_file(str(ROOT / "artifacts/tokenizer.json"))
    raw_tokens = sum(len(tokenizer.encode(message).ids) for message in MESSAGES)
    annotated_tokens = sum(len(tokenizer.encode(add_emoji_context(message)).ids) for message in MESSAGES)
    result = {
        "recognized_sequences": database.sequence_count,
        "max_emoji_version": database.max_emoji_version,
        "cold_database_build_ms": round(startup_ms, 3),
        "peak_build_memory_mib": round(peak / 1_048_576, 3),
        "preprocessing_messages": repeats * len(MESSAGES),
        "average_preprocessing_ms": round(elapsed * 1_000 / (repeats * len(MESSAGES)), 5),
        "sample_raw_tokens": raw_tokens,
        "sample_annotated_tokens": annotated_tokens,
        "sample_added_tokens": annotated_tokens - raw_tokens,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
