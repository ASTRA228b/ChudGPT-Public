"""Run a stateful Discord-style acceptance conversation against Public V20."""

from __future__ import annotations

import json
from pathlib import Path

from public_api_server import PublicModelService


def main() -> None:
    service = PublicModelService(Path("checkpoints/public_v20_final/best.pt"), "cuda")
    session = "discord-v20-held-out"
    context = "server=Astra Lab; channel=ai; speaker=Astra; relationship=ChudGPT developer Astra"
    prompts = (
        "Hello mate!",
        "hru rn",
        "What server are we in?",
        "Who am I?",
        "What does a server role do?",
        "What is the 67 meme?",
        'Say "yes, I hear you"',
        "234523242342 + 43242423",
        "Write a JavaScript function that rolls a six-sided die.",
        "Now explain that code simply.",
        "opentaiko",
        "Let's switch topics: tell me one fact about the Moon.",
    )
    transcript = []
    for prompt in prompts:
        _, reply = service.chat(prompt, session, 220, 0.58, "discord", context)
        transcript.append({"user": prompt, "chudgpt": reply})
        print(f"USER: {prompt}\nCHUDGPT: {reply}\n")
    Path("reports/v20_discord_conversation.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
