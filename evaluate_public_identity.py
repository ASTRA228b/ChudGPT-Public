"""Held-out benchmark for Public V20 AI, self, and model-family knowledge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from public_api_server import PublicModelService


CASES: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("ai", "What is AI?", ("artificial intelligence",), ()),
    ("ai", "Explain artificial intelligence to someone new to it.", ("computer", "learn"), ()),
    ("ai", "AI", ("artificial intelligence",), ()),
    ("ai", "How does AI work?", ("patterns", "trained"), ()),
    ("ai", "Does an LLM check every claim before saying it?", ("no", "verify"), ()),
    ("self", "Who are you?", ("chudgpt-public",), ("chatgpt", "openai")),
    ("self", "What is your full model name?", ("chudgpt-public",), ("chatgpt",)),
    ("self", "Tell me exactly which assistant is replying.", ("chudgpt-public",), ("chatgpt",)),
    ("self", "Tell me about yourself.", ("chudgpt-public", "language model"), ("chatgpt",)),
    ("self", "Say more about yourself and how you work.", ("chudgpt-public", "transformer"), ("chatgpt",)),
    ("self", "Are you a human being?", ("no", "ai"), ()),
    ("self", "Are you conscious and do you have feelings?", ("no",), ("yes",)),
    ("self", "Who created ChudGPT?", ("astra",), ("openai",)),
    ("family", "What is the ChudGPT model family?", ("public", "plus", "pro", "code", "music"), ()),
    ("family", "List every current and historical ChudGPT model you know.", ("buggy", "mega", "700", "1600"), ()),
    ("family", "What is ChudGPT-Public-Music V1?", ("music", "song"), ()),
    ("family", "What is ChudGPT Code for?", ("program",), ()),
    ("family", "How are ChudGPT Plus and Pro related?", ("plus", "pro"), ()),
    ("family", "What are Buggy ChudGPT and MEGA CHUD?", ("buggy", "mega", "chaotic"), ()),
    ("family", "Which ChudGPT was deliberately made chaotic?", ("mega", "chaotic"), ()),
    ("capability", "Can you browse the web by yourself?", ("no",), ("yes",)),
    ("capability", "What can you help me with?", ("chat", "code", "math"), ()),
)


def passes(reply: str, required: tuple[str, ...], forbidden: tuple[str, ...]) -> bool:
    lowered = reply.lower()
    return all(term in lowered for term in required) and not any(term in lowered for term in forbidden)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    parser.add_argument("--raw-neural", action="store_true", help="Bypass narrow factual grounding")
    args = parser.parse_args()
    service = PublicModelService(
        Path(args.checkpoint), args.device, tokenizer_path=Path(args.tokenizer)
    )
    results: list[dict[str, object]] = []
    for index, (category, prompt, required, forbidden) in enumerate(CASES):
        try:
            if args.raw_neural:
                reply = service._generate_raw(
                    [{"role": "user", "content": prompt}], 180, 0.58, service.system_prompt
                )
                service.last_assistance_reason = None
            else:
                _, reply = service.chat(prompt, f"identity-{index}", 180, 0.58)
            error = None
        except RuntimeError as exc:
            reply = ""
            error = str(exc)
        passed = passes(reply, required, forbidden)
        results.append({
            "category": category,
            "prompt": prompt,
            "reply": reply,
            "required": required,
            "forbidden": forbidden,
            "passed": passed,
            "assistance_reason": service.last_assistance_reason,
            "error": error,
        })
        print(f"[{category}] {'PASS' if passed else 'FAIL'} {reply}")
    categories = {}
    for category in sorted({str(result["category"]) for result in results}):
        group = [result for result in results if result["category"] == category]
        categories[category] = {
            "score": sum(bool(result["passed"]) for result in group),
            "total": len(group),
        }
    report = {
        "checkpoint": args.checkpoint,
        "raw_neural": args.raw_neural,
        "score": sum(bool(result["passed"]) for result in results),
        "total": len(results),
        "categories": categories,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SCORE {report['score']}/{report['total']}")


if __name__ == "__main__":
    main()
