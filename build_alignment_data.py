"""Build balanced response-type alignment data for ChudGPT-Public v4."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "public_conversations.jsonl"
OUTPUT = ROOT / "data" / "alignment_conversations.jsonl"
TARGET = 6_000


def record(*turns: tuple[str, str]) -> dict[str, object]:
    messages: list[dict[str, str]] = []
    for user, assistant in turns:
        messages.extend(({"role": "user", "content": user}, {"role": "assistant", "content": assistant}))
    return {"messages": messages, "source": "chudgpt-public-alignment-v4"}


def fingerprint(row: dict[str, object]) -> str:
    return json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)


def category(row: dict[str, object]) -> str:
    messages = row["messages"]
    assert isinstance(messages, list)
    users = " ".join(str(message["content"]) for message in messages if message["role"] == "user").lower()
    assistants = " ".join(str(message["content"]) for message in messages if message["role"] == "assistant").lower()
    if "```" in assistants or any(term in users for term in ("code", "python", "c#", "javascript", "unity")):
        return "code"
    if any(term in users for term in ("meme", "rickroll", "brainrot", "chill guy", "aura", "67")):
        return "meme"
    if len(messages) > 2:
        return "multiturn"
    if re.search(r"\d.*(?:[+*×÷/%]|mph|divided|cost|percent).*\d", users):
        return "math"
    if any(term in users for term in ("who are you", "what are you", "your name", "chudgpt", "what can you do")):
        return "identity"
    if users.startswith(("why", "explain", "how come")):
        return "explain"
    return "everyday"


def build(seed: int = 232) -> list[dict[str, object]]:
    rng = random.Random(seed)
    source = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    buckets: dict[str, list[dict[str, object]]] = {}
    for row in source:
        buckets.setdefault(category(row), []).append(row)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(row: dict[str, object]) -> None:
        key = fingerprint(row)
        if key not in seen:
            seen.add(key)
            row = {"messages": row["messages"], "source": "chudgpt-public-alignment-v4"}
            rows.append(row)

    # Select broad corpus examples with explicit per-category caps so arithmetic
    # cannot overwhelm ordinary language again.
    quotas = {"everyday": 900, "explain": 700, "meme": 650, "multiturn": 700, "math": 350, "identity": 100, "code": 100}
    for name, quota in quotas.items():
        choices = buckets.get(name, []).copy()
        rng.shuffle(choices)
        for row in choices[:quota]:
            add(row)

    capability_prompts = [
        "Hello! What can you do?", "Hey, what are you able to help with?", "What can we talk about?",
        "Give me the short version of your abilities.", "What sort of questions can I ask you?",
        "Hi. Tell me what you do.", "Are you useful for normal conversation?", "What are you good at?",
    ]
    capability_answers = [
        "I can have everyday conversations, explain basic facts, solve arithmetic and word problems, and help with simple code. I am small and experimental, so I can still make mistakes.",
        "You can chat with me, ask for an explanation, try math or logic, request a short story, or ask for beginner code and debugging help.",
        "I am useful for conversation, basic information, reasoning, math, and simple programming. I do not have live internet access.",
    ]
    for prompt in capability_prompts:
        for answer in capability_answers:
            for suffix in ("", " Please be honest.", " Keep this concise.", " I am just curious."):
                add(record((prompt + suffix, answer)))

    # Generative code examples vary both request and implementation details.
    variable_names = ["score", "total", "health", "speed", "count", "level", "coins", "distance"]
    for i in range(1_300):
        language = i % 5
        value = rng.randint(2, 500)
        variable = rng.choice(variable_names)
        if language == 0:
            prompt = f"Write Python code that stores {value} in {variable} and prints its double."
            answer = f"```python\n{variable} = {value}\nprint({variable} * 2)\n```"
        elif language == 1:
            sides = rng.randint(4, 20)
            prompt = f"Write a C# console program that rolls a {sides}-sided die. Return code only."
            answer = f"```csharp\nusing System;\n\nint roll = Random.Shared.Next(1, {sides + 1});\nConsole.WriteLine(roll);\n```"
        elif language == 2:
            prompt = f"Write JavaScript that adds {value} to every number in an array."
            answer = f"```javascript\nconst offset = {value};\nconst shifted = numbers.map(number => number + offset);\nconsole.log(shifted);\n```"
        elif language == 3:
            speed = rng.randint(1, 30)
            prompt = f"Write a Unity C# MonoBehaviour that moves a GameObject forward at speed {speed}."
            answer = f"```csharp\nusing UnityEngine;\n\npublic class ForwardMover : MonoBehaviour\n{{\n    [SerializeField] private float speed = {speed}f;\n\n    private void Update()\n    {{\n        transform.Translate(Vector3.forward * speed * Time.deltaTime);\n    }}\n}}\n```"
        else:
            prompt = f"Write a Python function that clamps a number between 0 and {value}."
            answer = f"```python\ndef clamp_value(number: float) -> float:\n    return max(0, min(number, {value}))\n```"
        add(record((prompt, answer)))

    # Debugging pairs teach diagnosis language without copying benchmark text.
    for i in range(650):
        error_kind = i % 5
        if error_kind == 0:
            prompt = f"My Python list has {i + 2} items and index {i + 2} fails. Why?"
            answer = f"Python uses zero-based indexes, so a list with {i + 2} items ends at index {i + 1}. Index {i + 2} is out of range."
        elif error_kind == 1:
            prompt = f"C# says the variable {rng.choice(variable_names)} might be unassigned. What does that mean?"
            answer = "The compiler found a path where the local variable could be read before receiving a value. Initialize it or assign it on every path before use."
        elif error_kind == 2:
            prompt = "Unity throws NullReferenceException when my script reads player.position. What should I check?"
            answer = "Check whether the player reference is assigned in the Inspector or found before use, and inspect the stack trace for the exact failing line."
        elif error_kind == 3:
            prompt = "My API request returned an HTML page instead of JSON. How should I debug it?"
            answer = "Check the status code, final URL, redirects, and Content-Type header. A wrong route or error page often returns HTML instead of JSON."
        else:
            prompt = "A loop never stops and freezes my program. What is the likely problem?"
            answer = "Its exit condition never becomes false, or it has no break. Trace the condition and make sure each iteration moves toward termination."
        add(record((prompt, answer)))

    # Novel multi-turn values teach references rather than a fixed teal example.
    names = ["Nova", "Atlas", "Miso", "Juniper", "Pixel", "Comet", "Maple", "Echo"]
    colors = ["amber", "cyan", "maroon", "violet", "lime", "navy", "coral", "silver"]
    objects = ["notebook", "backpack", "controller", "bicycle", "mug", "jacket"]
    for i in range(850):
        name, color, obj = rng.choice(names), rng.choice(colors), rng.choice(objects)
        if i % 3 == 0:
            add(record((f"My {obj} is {color}.", f"Got it—your {obj} is {color}."), (f"What color is my {obj}?", f"Your {obj} is {color}.")))
        elif i % 3 == 1:
            add(record((f"Call this plan {name} Harbor.", f"Okay—the plan is called {name} Harbor."), ("What did we call the plan?", f"We called it {name} Harbor.")))
        else:
            add(record(("Pick an animal for the logo.", f"I pick a {name.lower()} fox."), ("Why that one?", f"I picked the {name.lower()} fox because it could make a clear, energetic logo."), ("What did you pick?", f"I picked a {name.lower()} fox.")))

    # Generated logical relations provide held-out-style reasoning without using
    # the benchmark's names or sentences.
    shapes = ["circle", "triangle", "square", "star", "hexagon"]
    for i in range(700):
        name = rng.choice(names)
        shape = rng.choice(shapes)
        color = rng.choice(colors)
        add(record((f"Every {shape} in this puzzle is {color}. {name} is a {shape}. What color is {name}?", f"{name} is {color}, because every {shape} in the puzzle is {color}.")))

    rng.shuffle(rows)
    if len(rows) < TARGET:
        # Add further balanced corpus items, never unrestricted numeric rows.
        extras = buckets.get("everyday", []) + buckets.get("explain", []) + buckets.get("meme", []) + buckets.get("multiturn", [])
        rng.shuffle(extras)
        for row in extras:
            add(row)
            if len(rows) >= TARGET:
                break
    if len(rows) < TARGET:
        raise RuntimeError(f"Only {len(rows)} unique alignment rows were available")
    return rows[:TARGET]


def main() -> None:
    rows = build()
    OUTPUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows):,} balanced Public alignment conversations to {OUTPUT}")


if __name__ == "__main__":
    main()
