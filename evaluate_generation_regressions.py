"""Record repeatable neural-only answers to short tasks and follow-ups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from public_api_server import PublicModelService, selected_checkpoint


CASES = [
    ["Generate a simple for loop in Java."],
    ["Explain how a refrigerator works."],
    ["Compare TCP and UDP."],
    ["Write a Python function reversing a string."],
    ["Summarize: Bees pollinate flowers. Pollination helps plants reproduce."],
    ["My robot's name is Cedar.", "What is its name?"],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    service = PublicModelService(Path(args.checkpoint or selected_checkpoint()), "cuda")
    results = []
    for index, turns in enumerate(CASES):
        history = []
        for turn, prompt in enumerate(turns):
            torch.manual_seed(9100 + index * 10 + turn)
            history.append({"role": "user", "content": prompt})
            try:
                reply = service._generate_raw(history, 200, 0.6, service.system_prompt)
                error = None
            except RuntimeError as exc:
                reply, error = "", str(exc)
            results.append({"prompt": prompt, "reply": reply, "error": error})
            if reply:
                history.append({"role": "assistant", "content": reply})
            print(json.dumps(results[-1], ensure_ascii=True), flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
