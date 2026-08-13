"""Generate a fixed prompt-alignment report for comparable checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from chudlm.checkpoint import load_checkpoint
from chudlm.generation import generate
from chudlm.model import ModelConfig, TransformerLM
from chudlm.prompts import build_context_token_ids

CASES = [
    ("greeting", ["Hello!"], ("hello", "hey", "hi")),
    ("addition", ["What is 4 + 4?"], ("8",)),
    ("multiplication", ["What is 12 * 8?"], ("96",)),
    ("gravity", ["Explain gravity in one sentence."], ("mass", "attract")),
    ("ocean", ["What is the largest ocean on Earth?"], ("pacific",)),
    ("sky", ["What color is the sky on a clear day?"], ("blue",)),
    ("code_only", ["Write a Python function that adds two numbers. Code only."], ("def ", "return")),
    ("memory", ["My favorite color is teal.", "What color did I say was my favorite?"], ("teal",)),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device in {"cuda", "auto"} and torch.cuda.is_available() else "cpu")
    saved = load_checkpoint(Path(args.checkpoint), device)
    model = TransformerLM(ModelConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model"]); model.eval()
    tokenizer = Tokenizer.from_file("artifacts/tokenizer.json")
    eos = tokenizer.token_to_id("<eos>")
    results = []
    for name, prompts, expected in CASES:
        history = []
        replies = []
        for prompt in prompts:
            history.append({"role": "user", "content": prompt})
            _, ids = build_context_token_ids(tokenizer, history, model.config.context_length)
            output = generate(model, torch.tensor([ids], device=device), 100, 0.0, 50, 0.9, 1.1, eos)[0, len(ids):].tolist()
            reply = tokenizer.decode(output, skip_special_tokens=True).strip()
            replies.append(reply); history.append({"role": "assistant", "content": reply})
        text = replies[-1].lower()
        if name in {"addition", "multiplication"}:
            wanted = expected[0]
            numbers = re.findall(r"(?<!\d)-?\d+(?!\d)", text)
            passed = bool(numbers) and numbers[-1] == wanted
        else:
            passed = all(term in text for term in expected)
        results.append({"name": name, "prompts": prompts, "replies": replies, "passed": passed})
        print(f"[{name}] {'PASS' if passed else 'FAIL'}: {replies[-1]}")
    report = {"checkpoint": args.checkpoint, "passed": sum(x["passed"] for x in results), "total": len(results), "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Score: {report['passed']}/{report['total']}")


if __name__ == "__main__": main()
