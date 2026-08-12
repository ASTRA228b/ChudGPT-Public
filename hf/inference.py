"""Download or locally run ChudGPT-Public."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import ChudGPTPublic

SYSTEM_PROMPT = "You are ChudGPT, a small friendly and helpful conversational AI."


@torch.inference_mode()
def sample(model: ChudGPTPublic, ids: list[int], maximum: int, temperature: float, top_p: float, eos_id: int) -> list[int]:
    device = next(model.parameters()).device
    generated = list(ids)
    for _ in range(maximum):
        context = generated[-model.config.context_length:]
        logits = model(torch.tensor([context], device=device))[0, -1].float() / max(temperature, 1e-5)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        remove = cumulative - probabilities > top_p
        sorted_logits[remove] = float("-inf")
        filtered = torch.full_like(logits, float("-inf")).scatter(0, sorted_indices, sorted_logits)
        next_id = int(torch.multinomial(torch.softmax(filtered, dim=-1), 1).item())
        if next_id == eos_id:
            break
        generated.append(next_id)
    return generated[len(ids):]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--repo-id", help="Download a Hugging Face repo first, e.g. USER/ChudGPT-Public")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    directory = Path(args.model_dir)
    if args.repo_id:
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(args.repo_id))
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    model = ChudGPTPublic.from_pretrained(directory, device=device)
    eos_id = tokenizer.token_to_id("<eos>")
    history: list[tuple[str, str]] = []
    print(f"ChudGPT-Public loaded on {device}. Type /quit or /clear.")
    while True:
        user = input("You: ").strip()
        if user == "/quit":
            break
        if user == "/clear":
            history.clear()
            print("Conversation cleared.")
            continue
        if not user:
            continue
        turns = "\n".join(f"<user>: {question}\n<assistant>: {answer}" for question, answer in history)
        prompt = f"<system>: {SYSTEM_PROMPT}\n{turns}\n<user>: {user}\n<assistant>:"
        prompt_ids = tokenizer.encode(prompt).ids[-(model.config.context_length - args.max_new_tokens):]
        output = sample(model, prompt_ids, args.max_new_tokens, 0.7, 0.9, eos_id)
        answer = tokenizer.decode(output, skip_special_tokens=True).strip()
        print(f"ChudGPT: {answer}")
        history.append((user, answer))
        history = history[-8:]


if __name__ == "__main__":
    main()
