"""Build a larger, de-duplicated English foundation corpus for Public v18.

The earlier clean run used only the preferred assistant branch from OASST1.
That was useful for instruction quality but much too small for language
pretraining.  This builder uses every reviewed English message as foundation
text while keeping response-only SFT restricted to the preferred branches.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as parquet

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "data/downloads/oasst1-train.parquet"
OUTPUT = ROOT / "data/raw_v18/public_language.jsonl"


def clean_text(value: str) -> str | None:
    text = unicodedata.normalize("NFKC", value).replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    if not 20 <= len(text) <= 12_000:
        return None
    if "�" in text:
        return None
    # Identifier/hash-like samples teach token soup rather than language.
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) < 4 or sum(ch.isalpha() for ch in text) / max(1, len(text)) < 0.45:
        return None
    return text


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"OASST1 source not found: {SOURCE}")
    table = parquet.read_table(
        SOURCE,
        columns=["text", "lang", "deleted", "review_result", "tree_state"],
    ).to_pylist()
    unique: dict[str, str] = {}
    rejected = 0
    for row in table:
        if row["lang"] != "en" or row["deleted"] or row["review_result"] is False:
            rejected += 1
            continue
        text = clean_text(row["text"] or "")
        if text is None:
            rejected += 1
            continue
        key = re.sub(r"\s+", " ", text).casefold()
        unique.setdefault(key, text)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for text in unique.values():
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(unique):,} unique reviewed English documents; rejected {rejected:,} rows")


if __name__ == "__main__":
    main()
