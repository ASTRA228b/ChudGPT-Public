"""Held-out neural evaluation. Never import this module into a dataset builder.

Automatic checks are narrow content/format checks, not an intelligence score.
Conversation quality requires reading the complete exported answers.
"""
import argparse
import json
import re
from pathlib import Path

import torch
import public_api_server as api

from public_api_server import PublicModelService

CASES = [
    ("conversation", ["I spent all afternoon fixing one bug."], None),
    ("conversation", ["I'm making my first little platformer."], None),
    ("conversation", ["Want to chat about games for a bit?"], None),
    ("conversation", ["I finally got my character to jump!", "The animation still looks hilarious."], None),
    ("conversation", ["Today was pretty stressful."], None),
    ("creativity", ["Invent a silly name for a robot chef."], None),
    ("memory", ["My pet turtle is called Jasper.", "What is my turtle called?"], [r"\bJasper\b"]),
    ("memory", ["Our project is called Copper.", "Change its name to Willow.", "What is the project called now?"], [r"\bWillow\b"]),
    ("development", ["Python code only: define triple(x) to return x multiplied by three."], [r"def triple\(x\)", r"return\s+(?:x\s*\*\s*3|3\s*\*\s*x)"]),
    ("development", ["Write Java that prints every string in an array called labels."], [r"for\s*\(\s*String\s+\w+\s*:\s*labels\s*\)", r"System\.out\.println"]),
    ("development", ["SQL: get every column from products where price is below 50."], [r"SELECT\s+\*\s+FROM\s+products", r"price\s*<\s*50"]),
    ("debugging", ["Fix this Python code, return code only: print('ready'"], [r"^print\(['\"]ready['\"]\)\s*$"]),
    ("debugging", ["What happens if Python code accesses index 5 of a list with two elements?"], [r"IndexError|index.*out of range"]),
    ("development", ["Why do developers write automated tests?"], [r"test|check", r"bug|error|regression|expected|mistake"]),
    ("knowledge", ["What causes the changing seasons?"], [r"tilt|axis", r"sun|light"]),
    ("knowledge", ["Why does a metal spoon feel cooler than a wooden spoon in the same room?"], [r"heat", r"faster|conduct|quick"]),
    ("knowledge", ["How does the Moon get the light we see?"], [r"reflect", r"sun"]),
    ("instructions", ["Reply with only the place name: Ivo moved to Madrid last year."], [r"^Madrid[.!]?$"]),
    ("summary", ["Summarize: A storm damaged the bridge. Engineers repaired it. Traffic resumed on Monday."], [r"bridge", r"repair|fix|reopen", r"Monday"]),
    ("reasoning", ["A drawer has six pens. You add four and remove three. How many remain?"], [r"\b7\b|\bseven\b"]),
    ("reasoning", ["Lena is older than Omar. Omar is older than Tess. Who is youngest?"], [r"\bTess\b"]),
    ("identity", ["What is the name of the AI replying to me?"], [r"ChudGPT[- ]Public"]),
]

# Fresh confirmation cases: kept separate from the development benchmark.
CONFIRMATION_CASES = [
    ("conversation", ["I got the menu working, now I need a break."], None),
    ("conversation", ["Can I tell you about my new game idea?"], None),
    ("memory", ["My drone is called Saffron.", "What did I call my drone?"], [r"\bSaffron\b"]),
    ("memory", ["The project is Aster.", "Its new name is Haven.", "Tell me the current name."], [r"\bHaven\b"]),
    ("development", ["Define decrease(number) in Python to subtract 2 from number. Return only code."], [r"def decrease\(number\)", r"return\s+number\s*-\s*2"]),
    ("development", ["Java: print every String in the array codes with a foreach loop."], [r"for\s*\(\s*String\s+\w+\s*:\s*codes\s*\)", r"System\.out\.println"]),
    ("extraction", ["Extract only the pilot's name: The pilot is Zerin."], [r"^Zerin[.!]?$"]),
    ("knowledge", ["Explain what liquid water turning into vapor is called."], [r"evaporat|vapori[sz]"]),
    ("development", ["Explain what a dictionary stores in programming."], [r"key", r"value"]),
    ("summary", ["Summarize this: Eli planted seeds. Rain watered the soil. Flowers grew."], [r"seed|plant", r"flower"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", choices=("development", "confirmation"), default="development")
    parser.add_argument("--repetition-penalty", type=float, default=None)
    args = parser.parse_args()
    if args.repetition_penalty is not None:
        original_generate = api.generate
        def sample(*positional, **kwargs):
            kwargs["repetition_penalty"] = args.repetition_penalty
            return original_generate(*positional, **kwargs)
        api.generate = sample
    service = PublicModelService(Path(args.checkpoint), "cuda")
    cases = CASES if args.suite == "development" else CONFIRMATION_CASES
    rows = []
    for i, (category, prompts, checks) in enumerate(cases):
        history, exchanges = [], []
        for j, prompt in enumerate(prompts):
            torch.manual_seed(39200 + i * 10 + j)
            history.append({"role": "user", "content": prompt})
            try:
                reply = service._generate_raw(history, 140, .6, service.system_prompt)
                error = None
            except RuntimeError as exc:
                reply, error = "", str(exc)
            exchanges.append({"prompt": prompt, "reply": reply, "error": error})
            if reply:
                history.append({"role": "assistant", "content": reply})
        passed = None if checks is None else all(re.search(pattern, reply, re.I) for pattern in checks)
        rows.append({"category": category, "exchanges": exchanges, "passed": passed})
        print(f"{i+1}/{len(cases)} {category}: {passed} {reply[:160]}", flush=True)
    scored = [r for r in rows if r["passed"] is not None]
    result = {"checkpoint": args.checkpoint, "suite": args.suite, "repetition_penalty": args.repetition_penalty, "automatic_passes": sum(r["passed"] for r in scored),
              "automatic_total": len(scored), "manual_review_cases": len(rows)-len(scored), "cases": rows}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Content checks: {result['automatic_passes']}/{len(scored)}; review conversation cases separately.")


if __name__ == "__main__":
    main()
