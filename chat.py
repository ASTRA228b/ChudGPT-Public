from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from chudlm.checkpoint import load_checkpoint
from chudlm.generation import generate
from chudlm.model import ModelConfig, TransformerLM
from chudlm.prompts import build_context_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a trained ChudGPT checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "both"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device in {"cuda", "both"} and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} was requested but CUDA is not available")
    use_cuda = args.device in {"cuda", "both"} or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    checkpoint = load_checkpoint(Path(args.checkpoint), device)
    model = TransformerLM(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<eos>")
    messages: list[dict[str, str]] = []
    print(
        f"ChudGPT loaded from step {checkpoint.get('step', '?')} on {device}. "
        "Type /quit to exit, /clear to reset."
    )
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if user_text.lower() in {"/quit", "/exit"}:
            break
        if user_text.lower() == "/clear":
            messages.clear()
            print("ChudGPT: Conversation history cleared.")
            continue
        if not user_text:
            continue
        messages.append({"role": "user", "content": user_text})
        _, prompt_ids = build_context_token_ids(tokenizer, messages, model.config.context_length)
        generated = generate(
            model, torch.tensor([prompt_ids], device=device), args.max_new_tokens,
            args.temperature, args.top_k, args.top_p, args.repetition_penalty, eos_id,
        )[0, len(prompt_ids) :].tolist()
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"ChudGPT: {response}")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as error:
        print(f"Chat failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
