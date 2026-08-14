"""Build a turn-balanced Public-only corpus without hidden multi-turn skew."""

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
    ROOT.parent / "data/ultimate_sft/conversations.jsonl",
    ROOT.parent / "data/custom_sft/personality.jsonl",
    ROOT.parent / "data/demo_finetune/conversations.jsonl",
]
OUTPUT = ROOT / "data/public_v13_conversations.jsonl"
RAW_OUTPUT = ROOT / "data/raw_v13/public_corpus.jsonl"
QUOTAS = {
    "general": 5_000,
    "math": 1_000,
    "code": 1_400,
    "identity": 700,
    "meme": 250,
    "uncertainty": 300,
}
REPEAT_CAP = {
    "general": 8, "math": 2, "code": 8, "identity": 60,
    "meme": 15, "uncertainty": 20,
}
BAD = (
    "training references", "demo_topic_", "caption and conversation around it",
    "the main reason is that", "the surprising part of", "one useful way into",
    "there are several directions", "a good place to start", "generic introduction",
    "the secret number", "in the puzzle", "multiply quantity by price",
    "distance equals speed times time", "Ãƒ", "ï¿½", "Ã¢â‚¬",
)


def category_for(prompt: str) -> str:
    lowered = prompt.lower()
    if re.search(r"\b(chudgpt|who are you|what are you|your name|your identity|which assistant)\b", lowered):
        return "identity"
    if re.search(r"\b(meme|slang|cooked|aura|sahur|skibidi|rizz|brainrot|67|chud)\b", lowered):
        return "meme"
    if re.search(r"\b(don't know|do not know|uncertain|live internet|today's|current news|my dog|guess)\b", lowered):
        return "uncertainty"
    if re.search(r"\b(python|javascript|typescript|html|css|c\+\+|c#|unity|code|function|script|debug|api|sql|program)\b", lowered):
        return "code"
    if re.search(
        r"\b(calculate|solve|percent|factorial|prime|times|divided|sum|product|plus|minus|result|"
        r"split equally|how many|how much|total|average|difference|remainder|square root)\b|"
        r"\d\s*(?:[+*/%×÷]|-(?=\s*\d))",
        lowered,
    ):
        return "math"
    return "general"


def iter_pairs(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" not in row:
            yield str(row.get("instruction", "")).strip(), str(row.get("response", "")).strip()
            continue
        pending_user: str | None = None
        for message in row["messages"]:
            role = message.get("role")
            content = str(message.get("content", "")).strip()
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user is not None:
                yield pending_user, content
                pending_user = None


def main() -> None:
    rng = random.Random(1313)
    candidates: list[tuple[str, str]] = []
    for source in SOURCES:
        if source.is_file():
            candidates.extend(iter_pairs(source))
    rng.shuffle(candidates)
    buckets: dict[str, list[dict[str, object]]] = {name: [] for name in QUOTAS}
    repeats: Counter[str] = Counter()
    seen: set[str] = set()
    for user, assistant in candidates:
        text = f"{user}\n{assistant}"
        if not user or not assistant or any(fragment.lower() in text.lower() for fragment in BAD):
            continue
        fingerprint = re.sub(r"\s+", " ", text.casefold()).strip()
        if fingerprint in seen:
            continue
        category = category_for(user)
        if len(buckets[category]) >= QUOTAS[category] or repeats[assistant] >= REPEAT_CAP[category]:
            continue
        seen.add(fingerprint)
        repeats[assistant] += 1
        buckets[category].append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "source": f"public-v13-{category}",
        })
    selected = [row for category in QUOTAS for row in buckets[category]]
    rng.shuffle(selected)
    OUTPUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n", encoding="utf-8")
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT.write_text(OUTPUT.read_text(encoding="utf-8"), encoding="utf-8")
    counts = {category: len(rows) for category, rows in buckets.items()}
    print(f"Wrote {len(selected):,} turn-balanced examples; categories={counts}; max_repeat={max(repeats.values())}")


if __name__ == "__main__":
    main()
