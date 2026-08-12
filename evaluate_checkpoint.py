from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from chudlm.checkpoint import load_checkpoint
from chudlm.generation import generate
from chudlm.model import ModelConfig, TransformerLM
from chudlm.prompts import build_context_token_ids

PROMPTS = [
    "Hello!",
    "Hi there",
    "What is your name?",
    "Who made you?",
    "What is your purpose?",
    "Thanks",
    "What is 7 + 5?",
    "Explain a C# variable simply.",
    "Give me an idea for a Unity game.",
    "Explain why the sky is blue.",
    "Tell me a short joke.",
    "CHUD",
    "Are you Open Assistant?",
    "What language model are you?",
    "Goodbye",
    "Hey ChudGPT, what kinds of projects can you help me build?",
    "What is 19 + 8?",
    "What is 9 times 7?",
    "Why does the Moon appear to change shape?",
    "Show a tiny Python function that doubles a number.",
    "What is wrong with this Python code: if score > 5 print(score)",
    "Should physics movement go in Update or FixedUpdate in Unity?",
    "Explain HTML, CSS, and JavaScript in one short paragraph.",
    "Give me a two-step plan for debugging a crashing app.",
    "florble snax 992??",
    "Can you see what is currently open on my computer?",
    "Write two sentences about a lonely spaceship.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on fixed unseen prompts.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", default="reports/checkpoint_evaluation.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    saved = load_checkpoint(Path(args.checkpoint), device)
    model = TransformerLM(ModelConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<eos>")
    results: list[dict[str, str]] = []
    torch.manual_seed(42)
    for prompt in PROMPTS:
        _, prompt_ids = build_context_token_ids(
            tokenizer, [{"role": "user", "content": prompt}], model.config.context_length
        )
        output_ids = generate(
            model,
            torch.tensor([prompt_ids], device=device),
            max_new_tokens=64,
            temperature=0.0 if args.greedy else 0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=eos_id,
        )[0, len(prompt_ids) :].tolist()
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        results.append({"prompt": prompt, "response": response})
        print(f"\nPROMPT: {prompt}\nRESPONSE: {response}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "step": saved.get("step"),
                "generation": {
                    "temperature": 0.0 if args.greedy else 0.7,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                },
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved evaluation to {output_path}")


if __name__ == "__main__":
    main()
