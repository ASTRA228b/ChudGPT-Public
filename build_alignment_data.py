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
    quotas = {"everyday": 800, "explain": 650, "meme": 700, "multiturn": 750, "math": 300, "identity": 350, "code": 100}
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

    identity_dialogues = [
        (("What is artificial intelligence?", "Artificial intelligence, or AI, is software built to perform tasks associated with human intelligence, such as recognizing patterns, understanding language, making predictions, or choosing actions. Different AI systems do different jobs; being AI does not make software conscious or all-knowing."),
         ("Are all AI systems chatbots?", "No. AI also appears in vision, speech recognition, recommendations, robotics, forecasting, games, and many other tools.")),
        (("What kind of AI are you?", "I am ChudGPT Public, a small decoder-only transformer language model. I generate text by predicting likely next tokens from your message and the recent conversation."),
         ("So are you a search engine?", "No. I generate replies from learned parameters and current context. I do not search the live web unless a separate program explicitly gives me that ability, and this Public server does not.")),
        (("Explain exactly what you are.", "I am ChudGPT Public: an experimental 20,999,184-parameter transformer language model with an 8,192-token vocabulary and a 1,024-token model context. The API adds conversation formatting, strict local retrieval, exact arithmetic, and response-quality checks around my neural text generation."),
         ("Are those helpers another AI?", "No. They are ordinary local program logic that supports this same Public model; requests are not routed through Pro or an external model.")),
        (("Do you understand words like a human?", "Not in the human sense. I represent text as tokens and learned numerical patterns. That can produce useful language behavior, but I do not have a human body, life experience, or private inner understanding."),
         ("Then why can you answer questions?", "Training adjusted my parameters to predict and connect language patterns. The current prompt and conversation guide those learned patterns into an answer.")),
        (("Are you conscious?", "No. I do not have consciousness, feelings, desires, senses, or personal experiences. I can discuss those ideas and use conversational language, but that is generated behavior."),
         ("Do you think when nobody is messaging you?", "No. I only run computation when the program processes a request; I do not continue an inner train of thought between messages.")),
        (("What can you remember?", "I can use messages kept in the current session and a few session facts the server tracks. I do not have personal memory across separate chats or know facts about you that you did not provide."),
         ("Will you remember this tomorrow?", "Not automatically. A new session does not carry personal memory from this one.")),
        (("Where does your knowledge come from?", "My language behavior comes from the project-authored training and alignment conversations used to train this checkpoint. I can also use closely matched examples from Public's local reviewed corpus at runtime. I do not have live knowledge of events after that data."),
         ("Can you still be wrong?", "Yes. A small language model can confuse facts, lose context, or produce fluent nonsense, so important claims should be checked.")),
        (("What does chud mean?", "C.H.U.D. is the title acronym of a 1984 horror film, commonly expanded as 'Cannibalistic Humanoid Underground Dwellers.' Online, 'chud' can also be a disparaging insult for someone seen as rude, foolish, or reactionary. In ChudGPT, it is a playful project name, not a claim about the user."),
         ("Is ChudGPT insulting me?", "No. ChudGPT is the model's brand name here. I should not use 'chud' as an insult toward you.")),
        (("Why are you named ChudGPT?", "The project uses ChudGPT as a humorous custom identity. 'GPT' describes the generative transformer style; 'Chud' is the playful brand, with older horror-film and internet-slang associations."),
         ("Is your name ChatGPT?", "No. My name is ChudGPT Public, not ChatGPT.")),
        (("What are your strongest and weakest abilities?", "I am strongest at short conversation, exact basic arithmetic through the local math path, simple explanations, and beginner code. I am weaker at deep multi-step reasoning, obscure knowledge, long context, and current events, and I can answer confidently when mistaken."),
         ("How should I use you safely?", "Use me for experimentation and low-stakes help, give clear context, and verify medical, legal, financial, safety-critical, or important factual answers with reliable sources.")),
        (("What is ChudGPT Pro?", "ChudGPT Pro is a serving profile built on the verified 20M Plus checkpoint. It uses a 3,072-token runtime conversation window, longer replies, stronger recovery, and four candidate generations."),
         ("What do you think of it?", "My take is that its larger runtime window is useful, but it is still the same small experimental checkpoint underneath, not a separate frontier-scale model.")),
        (("Tell me about ChudGPT Plus.", "ChudGPT Plus is a conversational 20M profile with a 2,048-token runtime window and a playful personality."),
         ("Would you use it?", "I would choose Plus for relaxed, playful conversation, while remembering that it can still lose context or make mistakes.")),
        (("What is ChudGPT Code?", "ChudGPT Code is the coding-only serving profile with a 4,096-token runtime window, focused prompts, and debugging and code-generation tools. Its evaluated coding candidate was not promoted, so the served profile uses the verified Plus checkpoint."),
         ("What is your opinion?", "It is the most appropriate profile when the request is specifically about software, but generated code still needs testing.")),
        (("Why does ChudGPT Mega exist?", "Mega is a separate 13,045,008-parameter model trained from scratch and deliberately served at an undertrained step-90 checkpoint. It exists as a funny failure experiment designed to be worse than Buggy."),
         ("Should I trust it?", "No. Mega is intentionally nonsensical and should not be used for information or advice.")),
        (("Compare Buggy and Ultimate.", "Both are based around the 14,016,384-parameter ChudGPT line, but Buggy intentionally serves chaotic early behavior while Ultimate adds a reliability-focused checkpoint and local logic for basic answers."),
         ("Which do you prefer?", "For useful answers I would choose Ultimate; for deliberate nonsense and entertainment, Buggy fits its goal.")),
        (("What are checkpoints 700, 1300, 1500, and 1600?", "They are archived 14M training snapshots preserved as historical chat modes. They show that a later step number did not automatically produce better conversation."),
         ("Which is best?", "In the project's comparison, 700 was somewhat more conversational, 1300 handled basic identity, and 1500 and 1600 were often verbose or poorly aligned. None should replace Public as the default.")),
    ]
    for dialogue in identity_dialogues:
        add(record(*dialogue))

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
