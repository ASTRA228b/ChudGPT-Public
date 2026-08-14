"""Build an isolated, deduplicated corpus for Public v11 from Public-owned data."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ROOT / "data/public_v10_conversations.jsonl",
    ROOT / "data/alignment_conversations.jsonl",
    ROOT / "data/public_conversations.jsonl",
]
CHAT_OUTPUT = ROOT / "data/public_v11_conversations.jsonl"
RAW_OUTPUT = ROOT / "data/raw_v11/public_corpus.jsonl"
TARGET = 7_900
# Keep arithmetic useful without letting it hijack ordinary prompts.  These
# caps reflect the genuinely distinct Public-owned examples currently
# available rather than filling categories with duplicated templates.
QUOTAS = {"math": 1_400, "code": 850, "identity": 500, "meme": 150, "general": 5_000}
ASSISTANT_REPEAT_CAPS = {"math": 3, "code": 12, "identity": 120, "meme": 30, "general": 12}
BAD = (
    "Ã", "�", "â€", "one useful way into", "the surprising part of",
    "there are several directions", "a good place to start", "distance equals speed times time",
    "multiply quantity by price", "caption and conversation around it",
)


def main() -> None:
    rng = random.Random(1110)
    rows: list[dict[str, object]] = []
    for path in SOURCES:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    rng.shuffle(rows)
    buckets: dict[str, list[dict[str, object]]] = {name: [] for name in QUOTAS}
    seen: set[str] = set()
    assistant_counts: Counter[str] = Counter()
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        text = " ".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))
        if any(fragment.lower() in text.lower() for fragment in BAD):
            continue
        assistants = [str(message["content"]).strip() for message in messages if message.get("role") == "assistant"]
        if not assistants or any(not answer for answer in assistants):
            continue
        key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        user_text = " ".join(str(message["content"]) for message in messages if message.get("role") == "user").lower()
        if re.search(r"\b(chudgpt|who are you|what are you|which assistant|your identity)\b", user_text):
            category = "identity"
        elif re.search(r"\b(meme|slang|cooked|aura|sahur|skibidi|rizz|brainrot|67)\b", user_text):
            category = "meme"
        elif re.search(r"\b(python|javascript|typescript|html|css|c\+\+|c#|unity|code|function|script|debug|api|sql)\b", user_text):
            category = "code"
        elif re.search(r"\b(calculate|solve|percent|factorial|prime|times|divided|sum|product)\b|\d\s*[%+*/×-]", user_text):
            category = "math"
        else:
            category = "general"
        if any(assistant_counts[answer] >= ASSISTANT_REPEAT_CAPS[category] for answer in assistants):
            continue
        if len(buckets[category]) >= QUOTAS[category]:
            continue
        seen.add(key); assistant_counts.update(assistants)
        buckets[category].append({"messages": messages, "source": f"public-v11-{category}"})
    selected = [row for name in QUOTAS for row in buckets[name]]
    if len(selected) < TARGET:
        counts = {name: len(items) for name, items in buckets.items()}
        raise RuntimeError(f"Only {len(selected):,} quota rows; expected {TARGET:,}; {counts}")
    rng.shuffle(selected)
    CHAT_OUTPUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n", encoding="utf-8")
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT.write_text(CHAT_OUTPUT.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {len(selected):,} unique conversations; categories={ {k: len(v) for k,v in buckets.items()} }; max assistant repetition={max(assistant_counts.values())}")


if __name__ == "__main__":
    main()
