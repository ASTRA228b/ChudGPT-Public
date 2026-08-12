"""Hugging Face Gradio Space for authenticated ChudGPT-Public API calls."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "ASTRA228b/ChudGPT-Public")
MODEL_DIR = Path(snapshot_download(MODEL_REPO_ID))
sys.path.insert(0, str(MODEL_DIR))

from inference import SYSTEM_PROMPT, sample  # noqa: E402
from model import ChudGPTPublic  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = ChudGPTPublic.from_pretrained(MODEL_DIR, device=DEVICE)
TOKENIZER = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
EOS_ID = TOKENIZER.token_to_id("<eos>")


def chat(message: str, history: list[dict[str, str]] | None) -> str:
    """Generate one answer; exposed as the named `/chat` Space API endpoint."""
    if not message or not message.strip():
        return "Please send a message."
    turns: list[str] = []
    for item in (history or [])[-8:]:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            turns.append(f"<{role}>: {content}")
    prompt = f"<system>: {SYSTEM_PROMPT}\n" + "\n".join(turns) + f"\n<user>: {message.strip()}\n<assistant>:"
    prompt_ids = TOKENIZER.encode(prompt).ids[-864:]
    output_ids = sample(MODEL, prompt_ids, maximum=160, temperature=0.7, top_p=0.9, eos_id=EOS_ID)
    answer = TOKENIZER.decode(output_ids, skip_special_tokens=True).strip()
    return answer or "I could not generate a readable response."


demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="ChudGPT-Public 21M",
    description="Small experimental model. Verify important answers.",
    api_name="chat",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
