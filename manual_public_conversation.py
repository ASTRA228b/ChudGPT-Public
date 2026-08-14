"""Run and save an honest 30-turn conversation sample against the live Public API."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

PROMPTS = (
    "67", "67!", "hello", "yo", "bro", "what", "why", "no", "yes", "okay",
    "wait", "huh", "explain", "tell me more", "do it again", "nah", "lol", "123",
    "123!", "1 + 1", "what is 67", "is 67 prime", "pick a number", "what are you doing",
    "who are you", "what is ChudGPT-Public", "whats chudgpt", "bro what", "why tho",
    "Write a tiny C# method that doubles an integer.",
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results: list[dict[str, str]] = []
    for index, prompt in enumerate(PROMPTS):
        # Groups of five provide real follow-up context without turning all 30
        # unrelated topics into one intentionally poisoned conversation.
        session_id = f"public-v9-manual-{index // 5}"
        payload = json.dumps({"message": prompt, "session_id": session_id, "max_new_tokens": 100, "temperature": 0.6}).encode()
        request = urllib.request.Request("http://127.0.0.1:8010/api/chat", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            reply = json.load(response)["reply"]
        results.append({"user": prompt, "assistant": reply})
        print(f"User: {prompt}\nChudGPT-Public: {reply}\n")
    Path("reports/public_v9_manual_30.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
