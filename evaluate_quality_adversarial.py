"""Held-out neural quality comparison for Public V20 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chudlm.response_quality import assess_generated_reply, has_structured_list
from public_api_server import PublicModelService

PROMPTS = (
    "that meeting had cursed toaster energy",
    "bro the moon looked huge tonight",
    "my sandwich betrayed me",
    "explain gravity without making a list",
    "write a Rust function that reverses text",
    "answer yes or no: does Python use indentation?",
    "explain rain in one sentence",
    "give exactly 3 ideas for a tiny game",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    service = PublicModelService(Path(args.checkpoint), args.device, assistance_enabled=False)
    rows = []
    for index, prompt in enumerate(PROMPTS):
        try:
            _, reply = service.chat(prompt, f"adversarial-{index}", 150, 0.58)
            valid, reasons = assess_generated_reply(prompt, reply)
        except RuntimeError as error:
            reply, valid, reasons = f"ERROR: {error}", False, ("generation-error",)
        row = {"prompt": prompt, "reply": reply, "valid": valid, "reasons": list(reasons), "list": has_structured_list(reply)}
        rows.append(row)
        print(f"{'PASS' if valid else 'FAIL'} {prompt!r} -> {reply!r} {reasons}", flush=True)
    report = {"checkpoint": args.checkpoint, "passed": sum(row["valid"] for row in rows), "total": len(rows), "results": rows}
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SCORE {report['passed']}/{report['total']}")


if __name__ == "__main__":
    main()
