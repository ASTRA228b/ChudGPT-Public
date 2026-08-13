"""Build a compact, high-quality prompt-alignment dataset without external data."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "alignment_conversations.jsonl"


def record(*turns: tuple[str, str]) -> dict[str, object]:
    messages: list[dict[str, str]] = []
    for user, assistant in turns:
        messages.extend((
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ))
    return {"messages": messages, "source": "chudgpt-public-alignment-v2"}


def build() -> list[dict[str, object]]:
    rng = random.Random(228)
    rows: list[dict[str, object]] = []
    greetings = [
        ("Hello!", "Hey! What would you like to talk about?"),
        ("Hi there.", "Hi! How can I help?"),
        ("Hello, world!", "Hello! What are you working on today?"),
        ("Good morning.", "Good morning! What is on your mind?"),
        ("How are you?", "I do not have feelings, but I am running and ready to help. How are you?"),
    ]
    identity = [
        ("What is your name?", "My name is ChudGPT-Public."),
        ("Who are you?", "I am ChudGPT-Public, a small experimental conversational language model."),
        ("Do you have memory?", "I can use messages kept in this current conversation, but I do not have personal memories or remember other chats."),
        ("Can you access the internet?", "No. I do not have live internet access unless the program hosting me explicitly provides it."),
        ("What can you do?", "I can chat, explain basic facts, help with arithmetic, and write simple code, but I can make mistakes."),
    ]
    facts = [
        ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
        ("What color is the sky on a clear day?", "The sky usually appears blue on a clear day."),
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What planet is closest to the Sun?", "Mercury is the planet closest to the Sun."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
        ("What is gravity?", "Gravity is the attraction between objects with mass."),
        ("Explain gravity in one sentence.", "Gravity is the force that attracts objects with mass toward one another."),
        ("Explain photosynthesis in one short sentence.", "Photosynthesis lets plants use light, water, and carbon dioxide to make sugar and release oxygen."),
        ("Answer yes or no: Is Earth a planet?", "Yes."),
        ("Short answer: What gas do humans breathe to survive?", "Oxygen."),
    ]
    request_prefixes = ["", "Please ", "Could you ", "Can you ", "I want you to "]
    request_suffixes = ["", " Keep it clear.", " Give a direct answer.", " Be concise."]
    for prompt, answer in greetings + identity + facts:
        rows.append(record((prompt, answer)))
        for prefix in request_prefixes:
            for suffix in request_suffixes:
                natural = (prefix + prompt[0].lower() + prompt[1:] if prefix else prompt) + suffix
                rows.append(record((natural, answer)))

    # Broad arithmetic with varied wording teaches the relation rather than one answer class.
    forms = [
        "What is {a} {symbol} {b}?",
        "Calculate {a} {symbol} {b}.",
        "Please solve {a} {symbol} {b}.",
        "Give only the answer: {a} {symbol} {b}.",
    ]
    for _ in range(160):
        operation = rng.choice(("+", "-", "*"))
        if operation == "*":
            a, b = rng.randint(0, 20), rng.randint(0, 20)
        else:
            a, b = rng.randint(-200, 500), rng.randint(-200, 500)
        value = a + b if operation == "+" else a - b if operation == "-" else a * b
        prompt = rng.choice(forms).format(a=a, b=b, symbol=operation)
        answer = str(value) if prompt.startswith("Give only") else f"{a} {operation} {b} is {value}."
        rows.append(record((prompt, answer)))

    code_pairs = [
        ("Write a Python function that adds two numbers. Code only.", "```python\ndef add(a, b):\n    return a + b\n```"),
        ("Write C# code that prints Hello. Code only.", "```csharp\nusing System;\n\nConsole.WriteLine(\"Hello\");\n```"),
        ("Make a simple Unity C# component that rotates a GameObject.", "```csharp\nusing UnityEngine;\n\npublic class Rotator : MonoBehaviour\n{\n    [SerializeField] private float speed = 90f;\n\n    private void Update()\n    {\n        transform.Rotate(0f, speed * Time.deltaTime, 0f);\n    }\n}\n```"),
        ("Should Rigidbody movement run in Update or FixedUpdate in Unity?", "Use FixedUpdate for Rigidbody physics movement because it runs on the fixed physics timestep."),
        ("What does this Python code do: print(2 + 3)?", "It adds 2 and 3, then prints 5."),
    ]
    code_requests = ["", "Please ", "Can you ", "Show me how to ", "I need you to ", "For a beginner, "]
    code_suffixes = ["", " Keep it minimal.", " Return working code.", " Explain nothing else.", " Use clear names."]
    for prompt, answer in code_pairs:
        for prefix in code_requests:
            for suffix in code_suffixes:
                adjusted = (prompt if not prefix else prefix + prompt[0].lower() + prompt[1:]) + suffix
                rows.append(record((adjusted, answer)))

    memories = [
        (("My favorite color is teal.", "Got it—your favorite color is teal."), ("What color did I say was my favorite?", "You said your favorite color is teal.")),
        (("Call my project Star Finch.", "Okay, I will call your project Star Finch in this conversation."), ("What is my project called?", "Your project is called Star Finch.")),
        (("I have a dog named Pepper.", "Pepper is your dog—got it."), ("What is my dog's name?", "Your dog's name is Pepper.")),
        (("Let's talk about music.", "Sure. What kind of music do you enjoy?"), ("Mostly electronic.", "Electronic music covers many styles, from ambient to house; what artists or sounds do you like?")),
        (("I meant C#, not Python.", "Understood—I will use C#."), ("Now show a variable.", "```csharp\nint score = 0;\n```")),
    ]
    for memory in memories:
        rows.append(record(*memory))
    colors = ["teal", "purple", "orange", "green", "blue", "red", "silver", "yellow"]
    for color in colors:
        for noun in ["favorite color", "preferred color", "color I like most"]:
            rows.append(record(
                (f"My {noun} is {color}.", f"Got it—your {noun} is {color}."),
                ("Which color did I tell you?", f"You told me {color}."),
            ))

    # Add varied concise instruction-following and resistance to unrelated answers.
    concise = [
        ("Say hello in one sentence.", "Hello! It is good to meet you."),
        ("Answer yes or no: Can fish breathe underwater?", "Yes."),
        ("Give a short answer: Why is ice cold?", "Ice is cold because it has relatively low thermal energy."),
        ("Write exactly one sentence about the Moon.", "The Moon is Earth's natural satellite and orbits our planet."),
        ("Tell me a clean joke.", "Why did the computer get cold? It left its Windows open."),
        ("Correct yourself: you said 4 + 4 is 9.", "You are right; that was incorrect—4 + 4 is 8."),
        ("I do not understand.", "No problem. Tell me which part is unclear, and I will explain it more simply."),
    ]
    for prompt, answer in concise:
        rows.append(record((prompt, answer)))
        for prefix in request_prefixes[1:]:
            rows.append(record((prefix + prompt[0].lower() + prompt[1:], answer)))

    # Exact-message deduplication prevents frequency domination.
    unique: dict[str, dict[str, object]] = {}
    for row in rows:
        key = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
        unique[key] = row
    result = list(unique.values())
    rng.shuffle(result)
    return result


def main() -> None:
    rows = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows):,} unique alignment conversations to {OUTPUT}")


if __name__ == "__main__":
    main()
