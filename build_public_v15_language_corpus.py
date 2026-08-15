"""Build a broad language foundation plus balanced Public SFT corpus.

The external portion is the already-downloaded, reviewed English OASST1 export
(Apache-2.0).  Public-owned corrective turns are mixed in after strict
deduplication.  This builder never modifies its input datasets.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OASST = ROOT.parent / "data/oasst_finetune/conversations.jsonl"
PUBLIC = ROOT / "data/public_v13_conversations.jsonl"
CONVERSATIONS = ROOT / "data/public_v15_conversations.jsonl"
DOCUMENTS = ROOT / "data/raw_v15/public_language.jsonl"

BAD_FRAGMENTS = (
    "open assistant", "openassistant", "chatgpt", "as an ai language model",
    "language model developed by", "demo_topic_", "training references",
    "caption and conversation around it", "the secret number", "in the puzzle",
    "multiply quantity by price", "distance equals speed times time",
    "bouguereau died in 1905", "�", "Ãƒ", "Ã¯Â¿Â½",
)


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def valid_text(text: str, minimum: int, maximum: int) -> bool:
    if not minimum <= len(text) <= maximum:
        return False
    if any(fragment in text.casefold() for fragment in BAD_FRAGMENTS):
        return False
    if len(re.findall(r"[A-Za-z]{2,}", text)) < 2:
        return False
    control = sum(ord(char) < 32 and char not in "\n\t\r" for char in text)
    return control == 0


def load_pairs(path: Path, source_name: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages", [])
        if len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            continue
        user = str(messages[0].get("content", "")).strip()
        assistant = str(messages[1].get("content", "")).strip()
        if not valid_text(user, 3, 4_000) or not valid_text(assistant, 8, 5_000):
            continue
        # Long copied/cited answers are useful less often than direct dialogue
        # for this tiny model and can dominate whole context windows.
        if assistant.count("http") > 2 or assistant.count("References:") > 0:
            continue
        result.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "source": source_name,
            "license": row.get("license", "project-owned"),
        })
    return result


def main() -> None:
    candidates = load_pairs(OASST, "OpenAssistant/oasst1") + load_pairs(PUBLIC, "ChudGPT-Public")
    seen_pairs: set[str] = set()
    assistant_repeats: Counter[str] = Counter()
    selected: list[dict[str, object]] = []
    rejected_duplicates = 0
    rejected_repeats = 0
    for row in candidates:
        messages = row["messages"]
        user = str(messages[0]["content"])
        assistant = str(messages[1]["content"])
        pair_key = hashlib.sha256(f"{normalized(user)}\n{normalized(assistant)}".encode()).hexdigest()
        assistant_key = normalized(assistant)
        if pair_key in seen_pairs:
            rejected_duplicates += 1
            continue
        repeat_cap = 12 if row["source"] == "ChudGPT-Public" else 3
        if assistant_repeats[assistant_key] >= repeat_cap:
            rejected_repeats += 1
            continue
        seen_pairs.add(pair_key)
        assistant_repeats[assistant_key] += 1
        selected.append(row)

    # Stable hash ordering makes rebuilds exactly reproducible while avoiding
    # large blocks from one source.
    selected.sort(key=lambda row: hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())
    CONVERSATIONS.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENTS.parent.mkdir(parents=True, exist_ok=True)
    with CONVERSATIONS.open("w", encoding="utf-8") as handle, DOCUMENTS.open("w", encoding="utf-8") as docs:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            user = row["messages"][0]["content"]
            assistant = row["messages"][1]["content"]
            docs.write(json.dumps({"text": f"<user>: {user}\n<assistant>: {assistant}"}, ensure_ascii=False) + "\n")
    by_source = Counter(str(row["source"]) for row in selected)
    print(f"Wrote {len(selected):,} unique conversations: {dict(by_source)}")
    print(f"Rejected {rejected_duplicates:,} duplicate pairs and {rejected_repeats:,} over-repeated answers")


if __name__ == "__main__":
    main()

