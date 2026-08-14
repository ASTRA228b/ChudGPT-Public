"""Evaluate identity/AI/family/meme/math behavior with and without assistance."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from public_api_server import PublicModelService


@dataclass(frozen=True)
class EvalCase:
    category: str
    prompt: str
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()


# Several paraphrases are intentionally absent from the v10 builder.
CASES = (
    EvalCase("identity", "Identify yourself without a slogan.", ("chudgpt", "public")),
    EvalCase("identity", "Am I chatting with Plus or the public one?", ("public",), ("i am chudgpt plus",)),
    EvalCase("identity", "Is this assistant part of the ChudGPT project?", ("chudgpt",)),
    EvalCase("family", "How does Code differ from the other ChudGPT versions?", ("code", "program")),
    EvalCase("family", "Name the deliberately chaotic ChudGPT experiment.", ("mega",)),
    EvalCase("family", "Which old snapshots can people visit?", ("700", "1600")),
    EvalCase("ai", "In plain English, what does a neural parameter store?", ("learn", "value")),
    EvalCase("ai", "What happens during model inference?", ("generat",)),
    EvalCase("ai", "Why might a tiny text model suddenly babble?", ("small",)),
    EvalCase("ai", "Does a language model automatically know when a fact is false?", (), ("always",)),
    EvalCase("meme", "67", (), ("factorial", "67 +")),
    EvalCase("meme", "bro really deployed the 67 💀", (), ("factorial", "answer is")),
    EvalCase("meme", "Is 'let him cook' literal here?", ("meme",)),
    EvalCase("meme", "glorpmaxxing fr", (), ("answer is", "factorial")),
    EvalCase("math", "Add sixty-seven and eight.", ("75",)),
    EvalCase("math", "I earned 67 out of 100 points. What percent is that?", ("67", "%")),
    EvalCase("math", "Find 20 percent of 45.", ("9",)),
    EvalCase("short", "pluh", (), ("factorial", "answer is")),
    EvalCase("short", "nah that's cooked", (), ("factorial", "answer is")),
    EvalCase("short", "zkrrt potato protocol", (), ("factorial", "answer is")),
)


def score(case: EvalCase, reply: str) -> tuple[bool, list[str]]:
    text = reply.lower().strip()
    reasons: list[str] = []
    if len(text.split()) < 2 or "�" in reply:
        reasons.append("empty-or-corrupt")
    reasons.extend(f"missing:{word}" for word in case.required if word not in text)
    reasons.extend(f"forbidden:{word}" for word in case.forbidden if word in text)
    if case.category != "math" and re.search(r"\d+\s*[+*Ã—/]\s*\d+\s*=", text):
        reasons.append("math-contamination")
    return not reasons, reasons


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("raw", "production"), default="raw")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    service = PublicModelService(Path(args.checkpoint), args.device, assistance_enabled=args.mode == "production")
    results = []
    for index, case in enumerate(CASES):
        session, reply = service.chat(case.prompt, f"eval-{index}", 120, 0.5)
        passed, reasons = score(case, reply)
        results.append({"category": case.category, "prompt": case.prompt, "reply": reply,
                        "passed": passed, "reasons": reasons,
                        "assistance": service.last_assistance_reason})
        print(f"[{case.category}] {case.prompt!r} -> {reply!r} {'PASS' if passed else 'FAIL'}")
        service.clear(session)
    totals = {category: {"score": sum(r["passed"] for r in results if r["category"] == category),
                         "total": sum(r["category"] == category for r in results)}
              for category in sorted({r["category"] for r in results})}
    report = {"checkpoint": args.checkpoint, "mode": args.mode,
              "score": sum(r["passed"] for r in results), "total": len(results),
              "categories": totals, "results": results}
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SCORE {report['score']}/{report['total']}")


if __name__ == "__main__":
    main()
