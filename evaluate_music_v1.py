"""Run a repeatable, non-cherry-picked Music V1 generation benchmark."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from public_api_server import MusicModelService

PROMPTS = [
    "Write me a song about water",
    "Write me a song about ChudGPT",
    "Write me a song about yourself",
    "Write me a song about a broken keyboard",
    "Write me a song about a thunderstorm",
    "Write me a song about a microwave",
    "Write me a song about space",
    "Write me a song about nothing",
    "Write me a song about coding at 3 AM",
    "Write me a song about a robot learning to dance",
]


def lyric_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line.strip().lower()) for line in text.splitlines()
            if len(re.findall(r"[a-z0-9']+", line.lower())) >= 4 and not line.lstrip().startswith("[")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ChudGPT-Public-Music V1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    service = MusicModelService(args.checkpoint, args.device)
    cases: list[dict[str, object]] = []
    titles: list[str] = []
    styles: list[str] = []
    section_signatures: list[tuple[str, ...]] = []
    all_lines: list[str] = []
    prompt_overlaps: list[float] = []
    internal_repeats = 0
    malformed = 0
    relevance_scores: list[float] = []
    for case_index, prompt in enumerate(PROMPTS):
        samples: list[str] = []
        prompt_words = set(re.findall(r"[a-z]{4,}", prompt.lower()))
        for sample_index in range(args.samples):
            _, reply = service.chat(prompt, f"benchmark-{case_index}-{sample_index}", max_new_tokens=420)
            samples.append(reply)
            title = re.search(r"(?im)^title\s*:\s*([^\n]+)", reply)
            style = re.search(r"(?im)^style\s*:\s*([^\n]+)", reply)
            sections = tuple(re.findall(r"(?m)^\[([^]\n]+)\]", reply))
            if title:
                titles.append(title.group(1).strip())
            if style:
                styles.append(style.group(1).strip())
            section_signatures.append(sections)
            lines = lyric_lines(reply)
            all_lines.extend(lines)
            internal_repeats += sum(count - 1 for count in Counter(lines).values() if count > 1)
            reply_words = set(re.findall(r"[a-z]{4,}", reply.lower()))
            prompt_overlaps.append(len(prompt_words & reply_words) / max(len(prompt_words), 1))
            relevance_scores.append(service._topic_relevance(prompt, reply))
            malformed += int(not reply.strip() or bool(re.search(r"(?:\b\w\b\s+){5,}", reply)))
        cases.append({"prompt": prompt, "samples": samples})

    line_counts = Counter(all_lines)
    metrics = {
        "prompts": len(PROMPTS), "samples_per_prompt": args.samples,
        "outputs": len(PROMPTS) * args.samples,
        "unique_title_ratio": round(len(set(titles)) / max(len(titles), 1), 3),
        "unique_style_ratio": round(len(set(styles)) / max(len(styles), 1), 3),
        "unique_structure_ratio": round(len(set(section_signatures)) / max(len(section_signatures), 1), 3),
        "mean_prompt_content_overlap": round(sum(prompt_overlaps) / max(len(prompt_overlaps), 1), 3),
        "mean_topic_relevance": round(sum(relevance_scores) / max(len(relevance_scores), 1), 3),
        "topic_relevance_failures": sum(score < 0.08 for score in relevance_scores),
        "internal_repeated_lines": internal_repeats,
        "cross_output_repeated_lines": sum(count - 1 for count in line_counts.values() if count > 1),
        "malformed_outputs": malformed,
    }
    payload = {"checkpoint": str(args.checkpoint), "metrics": metrics, "cases": cases}
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
