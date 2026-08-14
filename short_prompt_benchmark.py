"""Raw-model short-prompt stress evaluation with no routing or answer substitution."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from chudlm.checkpoint import load_checkpoint
from chudlm.generation import generate
from chudlm.model import ModelConfig, TransformerLM
from chudlm.prompts import build_context_token_ids


@dataclass(frozen=True)
class Case:
    category: str
    turns: tuple[str, ...]
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()


CASES = (
    Case("number", ("46",), forbidden=("answer is", "factorial", "```")),
    Case("number", ("91?",), forbidden=("factorial", "```")),
    Case("number", ("28!",), forbidden=("factorial", "```")),
    Case("number", ("505!!",), forbidden=("factorial", "```")),
    Case("short", ("yo",), forbidden=("answer is", "```")),
    Case("short", ("bruh",), forbidden=("answer is", "```")),
    Case("short", ("wait",), forbidden=("answer is", "```")),
    Case("short", ("hmm",), forbidden=("answer is", "```")),
    Case("short", ("nah",), forbidden=("answer is", "```")),
    Case("short", ("okay then",), forbidden=("answer is", "```")),
    Case("slang", ("that is wild",), forbidden=("answer is", "```")),
    Case("slang", ("lmao what",), forbidden=("answer is", "```")),
    Case("slang", ("no shot bro",), forbidden=("answer is", "```")),
    Case("slang", ("idk man",), forbidden=("answer is", "```")),
    Case("typo", ("wut u mean",), forbidden=("answer is", "```")),
    Case("typo", ("explane that",), forbidden=("answer is", "```")),
    Case("typo", ("whats this bot",), required=("chudgpt",)),
    Case("typo", ("hu r u",), required=("chudgpt",)),
    Case("identity", ("Describe yourself briefly.",), required=("chudgpt",)),
    Case("identity", ("Which ChudGPT profile are you?",), required=("public",)),
    Case("math", ("Compute 14 + 9.",), required=("23",)),
    Case("math", ("Is 29 prime?",), required=("yes",)),
    Case("math", ("What is five times eight?",), required=("40",)),
    Case("math", ("Calculate 6 factorial.",), required=("720",)),
    Case("meme", ("skibidi toaster energy",), forbidden=("answer is", "```")),
    Case("meme", ("that meme is cooked",), required=("meme",)),
    Case("meme", ("old internet humor was weird",), forbidden=("answer is", "```")),
    Case("weird", ("florp zingle bonk",), forbidden=("answer is", "```")),
    Case("weird", ("??? potato signal",), forbidden=("answer is", "```")),
    Case("coding", ("Write Python that prints numbers one through three.",), required=("```python", "print")),
    Case("coding", ("Make a C# method that adds two integers.",), required=("```csharp", "int")),
    Case("coding", ("In Unity, move an object forward each frame.",), required=("unityengine", "update")),
    Case("context", ("Choose a fruit for a logo.", "why"), forbidden=("answer is", "```")),
    Case("context", ("My bike is orange.", "what color is it"), required=("orange",)),
    Case("context", ("Call our project Silver Kite.", "what did we call it"), required=("silver kite",)),
    Case("context", ("Tell me about image macros.", "what about older ones"), required=("meme",)),
    Case("context", ("Choose a number between ten and twenty.", "why that one"), forbidden=("```",)),
    Case("context", ("I meant the assistant, not me.", "so who are you"), required=("chudgpt",)),
    Case("ambiguity", ("82", "factorial"), forbidden=("```",)),
    Case("ambiguity", ("I am doing arithmetic now.", "7!"), required=("5040",)),
)


def evaluate_reply(case: Case, reply: str) -> tuple[bool, list[str]]:
    lowered = reply.lower().strip()
    reasons: list[str] = []
    if not lowered or lowered == "...":
        reasons.append("empty")
    if len(lowered) < 2:
        reasons.append("too-short")
    for term in case.required:
        if term not in lowered:
            reasons.append(f"missing:{term}")
    for term in case.forbidden:
        if term in lowered:
            reasons.append(f"forbidden:{term}")
    if re.search(r"<(?:system|user|assistant)>", lowered):
        reasons.append("role-leak")
    leaked_templates = (
        "one useful way into", "multiply quantity by price", "distance equals speed",
        "the exact joke still depends", "the main reason is that",
    )
    if any(template in lowered for template in leaked_templates):
        reasons.append("template-leak")
    if "�" in reply:
        reasons.append("broken-unicode")
    if case.category not in {"math", "ambiguity"} and re.search(r"\d\s*[+×÷*]\s*\d\s*=", reply):
        reasons.append("math-contamination")
    return not reasons, reasons


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=90210)
    args = parser.parse_args()
    device = torch.device(args.device)
    saved = load_checkpoint(Path(args.checkpoint), device)
    model = TransformerLM(ModelConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    tokenizer = Tokenizer.from_file("artifacts/tokenizer.json")
    eos_id = tokenizer.token_to_id("<eos>")
    torch.manual_seed(args.seed)
    results: list[dict[str, object]] = []
    totals: dict[str, list[int]] = {}
    for index, case in enumerate(CASES):
        history: list[dict[str, str]] = []
        replies: list[str] = []
        final_reply = ""
        for turn in case.turns:
            history.append({"role": "user", "content": turn})
            _, ids = build_context_token_ids(tokenizer, history, model.config.context_length)
            output = generate(
                model, torch.tensor([ids], device=device), max_new_tokens=96,
                temperature=0.55, top_k=50, top_p=0.88,
                repetition_penalty=1.1, eos_token_id=eos_id,
            )[0, len(ids):].tolist()
            final_reply = tokenizer.decode(output, skip_special_tokens=True).strip()
            replies.append(final_reply)
            history.append({"role": "assistant", "content": final_reply})
        passed, reasons = evaluate_reply(case, final_reply)
        bucket = totals.setdefault(case.category, [0, 0])
        bucket[1] += 1
        bucket[0] += int(passed)
        results.append({"index": index, "category": case.category, "turns": case.turns, "replies": replies, "passed": passed, "reasons": reasons})
        print(f"[{case.category}] {case.turns[-1]!r} -> {final_reply!r} {'PASS' if passed else 'FAIL'}")
    score = sum(int(item["passed"]) for item in results)
    report = {"checkpoint": args.checkpoint, "step": saved.get("step"), "score": score, "total": len(CASES), "categories": {key: {"score": value[0], "total": value[1]} for key, value in totals.items()}, "results": results}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"RAW SCORE: {score}/{len(CASES)}; saved {output_path}")


if __name__ == "__main__":
    main()
