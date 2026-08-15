"""Build the decontaminated broad-language foundation used by Public v17."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/public_v16_sft.jsonl"
CONVERSATIONS = ROOT / "data/public_v17_conversations.jsonl"
DOCUMENTS = ROOT / "data/raw_v17/public_language.jsonl"


def main() -> None:
    broad: list[dict[str, object]] = []
    project: list[dict[str, object]] = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        (broad if row.get("source") == "OpenAssistant/oasst1" else project).append(row)
    DOCUMENTS.parent.mkdir(parents=True, exist_ok=True)
    with DOCUMENTS.open("w", encoding="utf-8") as handle:
        for row in broad:
            user = row["messages"][0]["content"]
            assistant = row["messages"][1]["content"]
            handle.write(json.dumps({"text": f"<user>: {user}\n<assistant>: {assistant}"}, ensure_ascii=False) + "\n")
    # SFT learns broad dialogue first; the already-capped 70 exact project
    # identity/meme examples are retained only here, never in base pretraining.
    CONVERSATIONS.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in broad + project) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(broad):,} clean foundation documents and {len(broad) + len(project):,} SFT pairs")


if __name__ == "__main__":
    main()

