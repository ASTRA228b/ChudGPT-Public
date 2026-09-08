"""Compare checkpoints on basic Discord-style conversation and arithmetic."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from public_api_server import PublicModelService


CASES = (
    ("name_memory", ("My name is Nugget.", "What is my name?"), (r"\bNugget\b",)),
    ("color_memory", ("My favorite color is teal.", "What color did I say I like?"), (r"\bteal\b",)),
    ("rename_memory", ("Call my game Lantern 46.", "Rename it Harbor 19.", "What is its current name?"), (r"\bHarbor\s+19\b",)),
    ("math_definition", ("What is math?",), (r"number|quantity|shape|pattern", r"add|subtract|multiply|divide|measure|calculate|operation")),
    ("basic_addition", ("What is 2 + 2?",), (r"\b4\b",)),
    ("followup_addition", ("What is 2 + 2?", "Now what is 3 + 4?"), (r"\b7\b",)),
    ("word_addition", ("I have 5 apples and get 3 more. How many apples do I have?",), (r"\b8\b",)),
    ("ordinary_chat", ("I finished cleaning my room.",), (r"clean|room|nice|good|great|feel",)),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    service = PublicModelService(Path(args.checkpoint), "cuda")
    results = []
    for index, (name, prompts, checks) in enumerate(CASES):
        session_id = f"checkpoint-basics-{index}"
        exchanges = []
        reply = ""
        torch.manual_seed(88000 + index)
        for prompt in prompts:
            _, reply = service.chat(
                prompt,
                session_id,
                max_new_tokens=120,
                temperature=0.45,
                context_mode="discord",
            )
            exchanges.append({"prompt": prompt, "reply": reply})
        passed = all(re.search(pattern, reply, re.I) for pattern in checks)
        results.append({"name": name, "passed": passed, "exchanges": exchanges})
        print(f"{name:20} {'PASS' if passed else 'FAIL'} | {reply}", flush=True)

    report = {
        "checkpoint": args.checkpoint,
        "passed": sum(item["passed"] for item in results),
        "total": len(results),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SCORE {report['passed']}/{report['total']}")


if __name__ == "__main__":
    main()
