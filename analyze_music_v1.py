"""Analyze Music V1 training data, evaluations, and private Discord logs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable

from public_api_server import MusicModelService


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


def load_structured_logs(paths: list[Path]) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(record.get("prompt", ""))
            reply = str(record.get("reply", record.get("output", "")))
            if prompt and reply:
                samples.append((prompt, reply))
    return samples


def summarize(samples: list[tuple[str, str]]) -> dict[str, object]:
    replies = [reply for _, reply in samples if reply.strip()]
    line_counts = Counter(line for reply in replies for line in normalized_lines(reply))
    phrase_counts = Counter(phrase for reply in replies for phrase in set(phrases(reply)))
    titles = Counter(match.group(1).strip().lower() for reply in replies if (match := TITLE_RE.search(reply)))
    styles = Counter(match.group(1).strip().lower() for reply in replies if (match := STYLE_RE.search(reply)))
    repeated_line_outputs = sum(any(count > 1 for count in Counter(normalized_lines(reply)).values()) for reply in replies)
    malformed_sections = sum(bool("[" in reply and not SECTION_RE.search(reply)) for reply in replies)
    relevance_scores = [MusicModelService._topic_relevance(prompt, reply) for prompt, reply in samples if reply.strip()]
    relevance_failures = sum(score < 0.08 for score in relevance_scores)
    suspicious_terms = {"hallway", "static", "screen", "signal", "courage", "storm", "light", "room"}
    suspicious_nouns = Counter(
        word for reply in replies for word in WORD_RE.findall(reply.lower()) if word in suspicious_terms
    )
    endings = Counter(
        " ".join(WORD_RE.findall(line.lower())[-4:])
        for reply in replies for line in reply.splitlines() if len(WORD_RE.findall(line)) >= 4
    )
    bad_examples = [
        {"prompt": prompt, "reply": reply, "relevance": round(score, 3)}
        for (prompt, reply), score in zip(samples, relevance_scores) if score < 0.08
    ][:8]
    return {
        "outputs": len(replies),
        "average_words": round(sum(len(WORD_RE.findall(reply)) for reply in replies) / max(len(replies), 1), 2),
        "outputs_with_internal_repeated_lines": repeated_line_outputs,
        "malformed_section_outputs": malformed_sections,
        "unique_titles": len(titles),
        "unique_styles": len(styles),
        "duplicate_title_percentage": round((sum(titles.values()) - len(titles)) / max(sum(titles.values()), 1) * 100, 2),
        "duplicate_style_percentage": round((sum(styles.values()) - len(styles)) / max(sum(styles.values()), 1) * 100, 2),
        "malformed_structure_rate": round(malformed_sections / max(len(replies), 1) * 100, 2),
        "prompt_topic_relevance_failures": relevance_failures,
        "prompt_topic_relevance_failure_rate": round(relevance_failures / max(len(relevance_scores), 1) * 100, 2),
        "suspicious_nouns": suspicious_nouns.most_common(),
        "common_endings": endings.most_common(15),
        "bad_examples": bad_examples,
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
    parser.add_argument("--music-log", type=Path, action="append", default=[])
    args = parser.parse_args()
    payload = {
        "dataset": summarize(load_dataset(args.dataset)),
        "discord_music": summarize(load_discord_music(args.discord_log)),
        "structured_music_logs": summarize(load_structured_logs(
            args.music_log or [Path("logs/music/generations.jsonl"), Path("reports/music_v1_generations.jsonl")]
        )),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
