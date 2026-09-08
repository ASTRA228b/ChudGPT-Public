"""Held-out Python, C#, and Unity generation checks for Public V20."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from public_api_server import PublicModelService


CASES = [
    ("python", "Python code only: define cube(number) and return the cube.", (r"def\s+cube", r"\*\*\s*3|\*\s*number\s*\*\s*number")),
    ("python", "Python code only: reverse the order of words in text.", (r"split", r"reversed|\[::?-1\]", r"join")),
    ("python", "Write Python that keeps only positive values from numbers.", (r"for|filter", r">\s*0")),
    ("python", "Write a Python function returning the smaller of two numbers.", (r"def", r"min|if")),
    ("python", "Read data.json as UTF-8 JSON in Python.", (r"json", r"open", r"utf-8")),
    ("python", "Handle a possible ValueError while converting text to int in Python.", (r"try", r"except\s+ValueError", r"int\(")),
    ("python", "Write a Python dataclass named Point with float x and y.", (r"dataclass", r"class\s+Point", r"x:\s*float", r"y:\s*float")),
    ("python", "Write an async Python function that sleeps for two seconds.", (r"async\s+def", r"await", r"asyncio\.sleep\(2")),
    ("python", "Explain why a Python set is useful.", (r"unique|duplicate", r"member|lookup|contain")),
    ("python", "Fix this Python: for item in items print(item)", (r"for\s+item\s+in\s+items:", r"print\(item\)")),
    ("csharp", "C# code only: write IsAdult(int age), true at 18 or older.", (r"bool\s+IsAdult", r"age\s*>=\s*18")),
    ("csharp", "Use C# LINQ to order players by Score descending.", (r"OrderByDescending", r"Score")),
    ("csharp", "Write C# TryParse logic that produces an int from text.", (r"int\.TryParse", r"out")),
    ("csharp", "Write a C# class named Counter with Increment and Value.", (r"class\s+Counter", r"Increment", r"Value")),
    ("csharp", "Write a C# interface IHealable with Heal(int amount).", (r"interface\s+IHealable", r"Heal\s*\(int\s+amount\)")),
    ("csharp", "Write async C# to wait 500 milliseconds.", (r"async", r"await", r"Task\.Delay\(500")),
    ("csharp", "Read config.txt asynchronously in C#.", (r"File\.ReadAllTextAsync", r"config\.txt")),
    ("csharp", "Explain a C# nullable int.", (r"int\?|Nullable", r"null|no value")),
    ("csharp", "Fix C#: string name = 42;", (r"string\s+name\s*=\s*\"42\"|int\s+name\s*=\s*42")),
    ("unity", "Unity C#: rotate transform 90 degrees per second around Y in Update.", (r"Update", r"Rotate", r"90", r"deltaTime")),
    ("unity", "Unity C#: make a serialized float jumpForce field.", (r"SerializeField", r"float\s+jumpForce")),
    ("unity", "Unity C#: detect another collider entering a trigger.", (r"OnTriggerEnter", r"Collider")),
    ("unity", "Why should Unity input usually be read in Update?", (r"frame", r"input")),
    ("debugging", "C# list has three items. Why does list[3] fail?", (r"0", r"2", r"range|index")),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    service = PublicModelService(Path(args.checkpoint), "cuda")
    results = []
    for index, (category, prompt, checks) in enumerate(CASES):
        torch.manual_seed(88000 + index)
        try:
            reply = service._generate_raw([{"role": "user", "content": prompt}], 180, 0.45, service.system_prompt)
            error = None
        except RuntimeError as exc:
            reply, error = "", str(exc)
        passed = bool(reply) and all(re.search(pattern, reply, re.I) for pattern in checks)
        results.append({"category": category, "prompt": prompt, "reply": reply, "error": error, "passed": passed})
        print(f"{index + 1:02}/{len(CASES)} {category}: {passed} | {reply[:180]}", flush=True)
    report = {
        "checkpoint": args.checkpoint,
        "passes": sum(item["passed"] for item in results),
        "total": len(results),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Code checks: {report['passes']}/{report['total']}")


if __name__ == "__main__":
    main()
