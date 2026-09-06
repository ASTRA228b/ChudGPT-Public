"""Grade-spanning held-out benchmark for the deployed Public math path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from public_api_server import PublicModelService


CASES = (
    ("K-2", "1 + 1", "= 2"),
    ("K-2", "Calculate 47 - 19", "= 28"),
    ("3-5", "What is 12 times 9?", "= 108"),
    ("3-5", "Compute 144 / 12", "= 12"),
    ("3-5", "Evaluate (8 + 4) * 3", "= 36"),
    ("3-5", "A train travels 72 mph for 2.5 hours. How far does it travel?", "180 miles"),
    ("6-8", "3/4 + 5/6", "= 19/12"),
    ("6-8", "What is -12.5 + 7.75?", "= -4.75"),
    ("6-8", "Find 35 percent of 840", "= 294"),
    ("6-8", "Find the median of 4, 18, 7, 11, 2", "7"),
    ("6-8", "Solve for x: 7x - 9 = 40", "x = 7"),
    ("6-8", "A rectangle is 13 by 7. Find its area and perimeter.", "Area = 13 × 7 = 91"),
    ("9-12", "Calculate 3^7", "= 2187"),
    ("9-12", "What is the square root of 625?", "= 25"),
    ("9-12", "Solve x^2 - 9x + 20 = 0", "x = 5 or x = 4"),
    ("9-12", "Find the percent increase from 240 to 300", "25%"),
    ("9-12", "Find the GCD of 462 and 1071", "= 21"),
    ("9-12", "Find the LCM of 45 and 84", "= 1260"),
    ("9-12", "20 choose 4", "= 4845"),
    ("9-12", "Is 104729 a prime number?", "is prime"),
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    service = PublicModelService(Path(args.checkpoint), args.device)
    results = []
    for index, (grade, prompt, expected) in enumerate(CASES):
        _, reply = service.chat(prompt, f"math-{index}", 180, 0.55)
        passed = expected.lower() in reply.lower()
        results.append({"grade": grade, "prompt": prompt, "reply": reply, "expected": expected, "passed": passed})
        print(f"{grade:5} {'PASS' if passed else 'FAIL'} {prompt} -> {reply}", flush=True)
    report = {"passed": sum(item["passed"] for item in results), "total": len(results), "results": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SCORE {report['passed']}/{report['total']}")


if __name__ == "__main__":
    main()
