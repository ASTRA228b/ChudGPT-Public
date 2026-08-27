"""Generate a small, repeatable quality report for Music V1 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from public_api_server import MusicModelService


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ChudGPT-Public-Music V1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    service = MusicModelService(args.checkpoint, args.device)
    cases = [
        ["Make a love song"],
        ["Make music"],
        ["Write a full dark electronic song about my WiFi dying"],
        ["Write a full song about nothing"],
        [
            "I want a nostalgic synth-pop song about the last summer night",
            "Write the full original lyrics and keep the style and title",
            "What style and song name did we choose?",
        ],
    ]
    results: list[dict[str, object]] = []
    for turns in cases:
        session_id: str | None = None
        transcript: list[dict[str, str]] = []
        for prompt in turns:
            session_id, reply = service.chat(
                prompt,
                session_id,
                max_new_tokens=420,
                temperature=0.72,
            )
            transcript.extend((
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply},
            ))
        results.append({"turns": transcript})

    payload = {"checkpoint": str(args.checkpoint), "cases": results}
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
