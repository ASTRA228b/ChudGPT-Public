"""Analyze Music V1 training data, evaluations, and private Discord logs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable


SECTION_RE = re.compile(r"(?m)^\[([^]\n]{2,40})\]\s*$")
TITLE_RE = re.compile(r"(?im)^title\s*:\s*(.+)$")
STYLE_RE = re.compile(r"(?im)^style\s*:\s*(.+)$")
WORD_RE = re.compile(r"[a-z0-9']+")


def normalized_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line.strip().lower()) for line in text.splitlines()
            if len(WORD_RE.findall(line)) >= 4 and not line.lstrip().startswith("[")]


def phrases(text: str, minimum: int = 3, maximum: int = 6) -> Iterable[str]:
    words = WORD_RE.findall(text.lower())
    for size in range(minimum, maximum + 1):
        for index in range(len(words) - size + 1):
            yield " ".join(words[index:index + size])


def load_dataset(path: Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record.get("messages", [])
        prompt = next((item.get("content", "") for item in messages if item.get("role") == "user"), "")
        for item in messages:
            if item.get("role") == "assistant":
                samples.append((prompt, item.get("content", "")))
    return samples


def load_discord_music(path: Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    if not path.is_file():
        return samples
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = str(record.get("prompt", ""))
        if re.match(r"(?i)^music(?:\s|$)", prompt):
            samples.append((re.sub(r"(?i)^music\s*", "", prompt), str(record.get("reply", ""))))
    return samples


def summarize(samples: list[tuple[str, str]]) -> dict[str, object]:
    replies = [reply for _, reply in samples if reply.strip()]
    line_counts = Counter(line for reply in replies for line in normalized_lines(reply))
    phrase_counts = Counter(phrase for reply in replies for phrase in set(phrases(reply)))
    titles = Counter(match.group(1).strip().lower() for reply in replies if (match := TITLE_RE.search(reply)))
    styles = Counter(match.group(1).strip().lower() for reply in replies if (match := STYLE_RE.search(reply)))
    repeated_line_outputs = sum(any(count > 1 for count in Counter(normalized_lines(reply)).values()) for reply in replies)
    malformed_sections = sum(bool("[" in reply and not SECTION_RE.search(reply)) for reply in replies)
    return {
        "outputs": len(replies),
        "average_words": round(sum(len(WORD_RE.findall(reply)) for reply in replies) / max(len(replies), 1), 2),
        "outputs_with_internal_repeated_lines": repeated_line_outputs,
        "malformed_section_outputs": malformed_sections,
        "unique_titles": len(titles),
        "unique_styles": len(styles),
        "worst_repeated_lines": line_counts.most_common(20),
        "worst_repeated_phrases": [(phrase, count) for phrase, count in phrase_counts.most_common(40) if count > 1][:20],
        "worst_repeated_titles": [(value, count) for value, count in titles.most_common(15) if count > 1],
        "worst_repeated_styles": [(value, count) for value, count in styles.most_common(15) if count > 1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/music_v1_conversations.jsonl"))
    parser.add_argument("--discord-log", type=Path, default=Path(r"D:\ChudGPT-Discord-Logs\discord-2026-08.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/music_v1_repetition_analysis.json"))
    args = parser.parse_args()
    payload = {
        "dataset": summarize(load_dataset(args.dataset)),
        "discord_music": summarize(load_discord_music(args.discord_log)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
