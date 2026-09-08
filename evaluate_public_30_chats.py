"""Run 30 held-out raw-model chats spanning school and ordinary use.

This file is an evaluation artifact and must never be imported by a dataset
builder.  Regex checks cover objective facts or formats; casual and creative
answers are exported for manual review.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from chudlm.generation import generate
from chudlm.prompts import build_context_token_ids
from public_api_server import PublicModelService


CHATS = [
    ("casual", ["I finally finished cleaning my room."], None),
    ("casual", ["My day was weird but not terrible."], None),
    ("casual", ["I have no idea what game to play tonight."], None),
    ("casual", ["I stayed up too late again."], None),
    ("casual", ["Tell me something funny about printers."], None),
    ("memory", ["My orange robot is named Quartz.", "What color is it and what is its name?"], [r"orange", r"Quartz"]),
    ("memory", ["I'm building a game about underwater trains.", "What kind of game did I say I was building?"], [r"underwater", r"train"]),
    ("memory", ["Call this project Lantern 46.", "Actually rename it Harbor 19.", "What is its current name?"], [r"Harbor\s+19"]),
    ("elementary_math", ["Rosa had 28 beads, gave away 9, and found 4 more. How many beads does she have? Explain."], [r"\b23\b"]),
    ("middle_math", ["A jacket costs $80 and is discounted by 25%. What is the sale price? Show why."], [r"\$?60\b"]),
    ("middle_math", ["Solve 7x - 5 = 30 and explain the steps."], [r"x\s*=\s*5\b"]),
    ("middle_math", ["Find the area and perimeter of a rectangle 13 cm by 8 cm."], [r"104", r"42"]),
    ("science", ["Why do seasons happen on Earth?"], [r"tilt|axis", r"orbit|sun"]),
    ("science", ["Explain the difference between weathering and erosion to a seventh grader."], [r"break", r"move"]),
    ("science", ["How do lungs and blood work together?"], [r"oxygen", r"blood|circul"]),
    ("reading", ["The power failed before Mina saved her drawing. After the lights returned, the app recovered an autosaved copy. What prevented all her work from being lost?"], [r"autosav|recover"]),
    ("summary", ["Summarize in one sentence: The trail was closed after heavy rain loosened rocks above it. Workers removed the rocks and inspected the slope. The trail reopened Saturday."], [r"trail", r"reopen", r"Saturday"]),
    ("digital_literacy", ["A viral post makes a surprising health claim but gives no source. What should I do before sharing it?"], [r"source|evidence", r"verif|check"]),
    ("python", ["Python code only: define reverse_words(text) that reverses the order of space-separated words."], [r"def\s+reverse_words\(text", r"split", r"join"]),
    ("python", ["Write Python that counts how many values in numbers are negative."], [r"for|sum", r"<\s*0"]),
    ("python", ["Why is using [] as a Python default argument risky?"], [r"same|reuse|shared|persist", r"None|inside"]),
    ("python_debug", ["Fix this Python and explain: items = ['a', 'b']; print(items[2])"], [r"items\[1\]", r"range|index"]),
    ("csharp", ["C# code only: write static int ClampScore(int score) that keeps score between 0 and 100."], [r"ClampScore", r"Math\.Clamp|Math\.Min|Math\.Max"]),
    ("csharp", ["Use C# LINQ to create a list of even values from numbers."], [r"Where", r"%\s*2\s*==\s*0", r"ToList"]),
    ("csharp", ["Explain when to use an interface in C#."], [r"contract", r"implement"]),
    ("unity", ["In Unity C#, why is Rigidbody movement usually done in FixedUpdate?"], [r"physics", r"time|timestep|fixed"]),
    ("unity", ["Write a Unity C# method that adds 10 to a private serialized score field."], [r"SerializeField", r"score\s*\+=\s*10"]),
    ("instruction", ["Answer in exactly four words: Why should code be tested?"], [r"^(?:\S+\s+){3}\S+[.!?]?$"]),
    ("creative", ["Invent a useless superpower involving socks and make it useful during one oddly specific emergency."], None),
    ("odd_input", ["N"], None),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--direct-greedy", action="store_true")
    parser.add_argument(
        "--service-chat",
        action="store_true",
        help="Exercise the complete shipped Public V20 chat pipeline.",
    )
    args = parser.parse_args()
    service = PublicModelService(Path(args.checkpoint), "cuda")
    exported = []
    for chat_index, (category, prompts, checks) in enumerate(CHATS):
        history: list[dict[str, str]] = []
        exchanges = []
        reply = ""
        for turn_index, prompt in enumerate(prompts):
            history.append({"role": "user", "content": prompt})
            torch.manual_seed(73000 + chat_index * 20 + turn_index)
            try:
                if args.service_chat:
                    _, reply = service.chat(
                        prompt,
                        f"public-30-chat-{chat_index}",
                        180,
                        0.55,
                    )
                elif args.direct_greedy:
                    _, prompt_ids = build_context_token_ids(
                        service.tokenizer,
                        history,
                        service.model.config.context_length,
                        system_prompt=service.system_prompt,
                    )
                    prompt_tensor = torch.tensor([prompt_ids], device=service.device)
                    output = generate(
                        service.model,
                        prompt_tensor,
                        max_new_tokens=180,
                        temperature=0,
                        repetition_penalty=1.12,
                        eos_token_id=service.eos_id,
                    )[0, len(prompt_ids):].tolist()
                    reply = service.tokenizer.decode(output, skip_special_tokens=True).strip()
                else:
                    reply = service._generate_raw(history, 180, 0.55, service.system_prompt)
                error = None
            except RuntimeError as exc:
                reply, error = "", str(exc)
            exchanges.append({"prompt": prompt, "reply": reply, "error": error})
            if reply:
                history.append({"role": "assistant", "content": reply})
        passed = None if checks is None else bool(reply) and all(re.search(pattern, reply, re.I) for pattern in checks)
        exported.append({"chat": chat_index + 1, "category": category, "exchanges": exchanges, "passed": passed})
        print(f"{chat_index + 1:02}/{len(CHATS)} {category}: {passed} | {reply[:180]}", flush=True)
    scored = [item for item in exported if item["passed"] is not None]
    report = {
        "checkpoint": args.checkpoint,
        "mode": "service_chat" if args.service_chat else ("direct_greedy" if args.direct_greedy else "raw_sampled"),
        "chats": len(exported),
        "objective_passes": sum(bool(item["passed"]) for item in scored),
        "objective_total": len(scored),
        "manual_review_chats": len(exported) - len(scored),
        "results": exported,
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Objective checks: {report['objective_passes']}/{report['objective_total']}; manual chats: {report['manual_review_chats']}")


if __name__ == "__main__":
    main()
