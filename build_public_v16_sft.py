"""Build a low-contamination SFT curriculum from broad reviewed dialogue."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/public_v15_conversations.jsonl"
OUTPUT = ROOT / "data/public_v16_sft.jsonl"

TEMPLATE_PATTERNS = (
    r"i am with you on the new topic",
    r"tell me what brought it to mind",
    r"what direction (?:are we|interests you)",
    r"we can keep \w+ casual",
    r"i chose a playful label for option",
    r"caption and conversation around it",
    r"the exact joke still depends",
    r"there are (?:a few|several) (?:ways|directions)",
    r"one useful way into",
    r"the surprising part of",
    r"the main reason is that cha",
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def main() -> None:
    broad: list[dict[str, object]] = []
    public_special: list[dict[str, object]] = []
    repeats: Counter[str] = Counter()
    seen: set[str] = set()
    rejected_templates = 0
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        user = str(row["messages"][0]["content"]).strip()
        assistant = str(row["messages"][1]["content"]).strip()
        combined = f"{user}\n{assistant}"
        if any(re.search(pattern, combined, re.IGNORECASE) for pattern in TEMPLATE_PATTERNS):
            rejected_templates += 1
            continue
        key = hashlib.sha256(normalize(combined).encode()).hexdigest()
        response_key = normalize(assistant)
        if key in seen or repeats[response_key] >= 2:
            continue
        seen.add(key)
        repeats[response_key] += 1
        if row.get("source") == "OpenAssistant/oasst1":
            broad.append(row)
            continue
        lowered = user.casefold()
        if re.search(r"\b(chudgpt|who are you|what are you|your name|which assistant)\b", lowered):
            public_special.append(row)
        elif re.search(r"\b(meme|slang|sahur|skibidi|rizz|brainrot|aura|cooked)\b", lowered):
            public_special.append(row)
    # Keep exact project knowledge present but unable to dominate broad syntax.
    public_special.sort(key=lambda row: hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())
    selected = broad + public_special[:300]
    selected.sort(key=lambda row: hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())
    OUTPUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected):,} SFT pairs ({len(broad):,} broad + {min(300, len(public_special)):,} Public-special)")
    print(f"Rejected {rejected_templates:,} template-family rows")


if __name__ == "__main__":
    main()

