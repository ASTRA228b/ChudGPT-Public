"""Build clean, varied Public v9 SFT data focused on raw conversational behavior."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from short_prompt_benchmark import CASES

ROOT = Path(__file__).resolve().parent
SOURCES = (ROOT / "data/public_conversations.jsonl", ROOT / "data/alignment_conversations.jsonl")
OUTPUT = ROOT / "data/public_v9_conversations.jsonl"
TARGET = 9_000
BAD_TEXT = ("�", "Ã", "Â", "one useful way into", "the exact joke still depends", "multiply quantity by price", "distance equals speed times time")
HELDOUT = {turn.strip().lower() for case in CASES for turn in case.turns}


def row(*pairs: tuple[str, str]) -> dict[str, object]:
    messages: list[dict[str, str]] = []
    for user, assistant in pairs:
        messages.extend(({"role": "user", "content": user}, {"role": "assistant", "content": assistant}))
    return {"messages": messages, "source": "chudgpt-public-v9-raw-model"}


def clean_record(candidate: dict[str, object]) -> bool:
    messages = candidate.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    text = " ".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))
    lowered = text.lower()
    return bool(text.strip()) and not any(bad.lower() in lowered for bad in BAD_TEXT)


def build(seed: int = 914) -> list[dict[str, object]]:
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    answer_counts: Counter[str] = Counter()

    def add(candidate: dict[str, object], answer_cap: int = 3) -> None:
        if not clean_record(candidate):
            return
        messages = candidate["messages"]
        assert isinstance(messages, list)
        if any(message["role"] == "user" and message["content"].strip().lower() in HELDOUT for message in messages):
            return
        answers = [str(message["content"]) for message in messages if message["role"] == "assistant"]
        key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        if key in seen or any(answer_counts[answer] >= answer_cap for answer in answers):
            return
        seen.add(key)
        records.append({"messages": messages, "source": "chudgpt-public-v9-raw-model"})
        answer_counts.update(answers)

    # Retain broad, clean knowledge/code/math examples without known leaked templates.
    source_rows: list[dict[str, object]] = []
    for path in SOURCES:
        source_rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    rng.shuffle(source_rows)
    for candidate in source_rows:
        add(candidate, answer_cap=2)
        if len(records) >= 4_400:
            break

    acknowledgements = [
        "Okay, I am with you.", "Got it. Keep going.", "Yeah, I hear you.",
        "Fair enough.", "That makes sense.", "All right. What happened next?",
        "I am following.", "Honestly, that is pretty funny.", "Wait, what happened?",
        "That is wild. Tell me more.", "No worries.", "I get what you mean.",
    ]
    short_inputs = ["yo", "bro", "bruh", "wait", "hmm", "nah", "yep", "okay", "lol", "lmao", "wild", "seriously", "no way", "idk", "maybe", "sure"]
    for prompt in short_inputs:
        for answer in rng.sample(acknowledgements, 4):
            add(row((prompt, answer)), answer_cap=4)

    # Bare numbers are conversational objects, not arithmetic unless context says so.
    reactions = [
        "You dropped the number {n}. What are we doing with it?",
        "{n}? Is that a choice, a score, or something else?",
        "I see {n}. Give me the context and I will follow.",
        "The mysterious {n} has entered the chat. What does it mean here?",
    ]
    for number in rng.sample(range(10, 990), 180):
        punctuation = rng.choice(("", "?", "!", "!!", "..."))
        add(row((f"{number}{punctuation}", rng.choice(reactions).format(n=number))), answer_cap=2)
    for number in rng.sample(range(3, 13), 8):
        add(row((f"Calculate {number} factorial.", f"{number} factorial is " + str(__import__("math").factorial(number)) + ".")))
        add(row(("We are doing factorials.", "Okay, we are working with factorials."), (f"{number}!", f"{number}! is {__import__('math').factorial(number)}.")))

    # Multi-turn short follow-ups with varied entities and answers.
    colors = ["teal", "amber", "violet", "coral", "navy", "silver", "lime", "maroon"]
    objects = ["bike", "mug", "jacket", "controller", "notebook", "backpack"]
    subjects = ["memes", "music", "space", "games", "movies", "animals"]
    for index in range(520):
        color, obj, subject = rng.choice(colors), rng.choice(objects), rng.choice(subjects)
        kind = index % 6
        if kind == 0:
            add(row((f"My {obj} is {color}.", f"Got it, your {obj} is {color}."), ("what color is it", f"Your {obj} is {color}.")))
        elif kind == 1:
            add(row((f"Pick a color for the {obj}.", f"I pick {color}."), ("why", f"I picked {color} because it would stand out clearly on the {obj}.")))
        elif kind == 2:
            add(row((f"Tell me something about {subject}.", f"{subject.title()} can be a huge topic, but a good starting point is how people remix familiar ideas into something personal."), ("tell me more", f"With {subject}, context matters: the same idea can feel completely different depending on the audience, style, and time period.")))
        elif kind == 3:
            add(row((f"I like {subject}.", f"Nice. What do you enjoy most about {subject}?"), ("the old stuff", f"Older {subject} can be fun because you can see which ideas lasted and which ones belonged to their moment.")))
        elif kind == 4:
            add(row(("I meant you, not me.", "Got it—you were asking about me."), ("so answer it", "I am ChudGPT-Public, a small experimental language model built for conversation, basic facts, math, and code.")))
        else:
            add(row((f"Let us talk about {subject}.", f"Sure, let us switch to {subject}. What part interests you?"), ("actually never mind", "No problem. We can change topics whenever you want.")))

    slang_starts = ["bro what", "nah man", "why tho", "huh???", "wat", "wut", "that is cooked", "this goes hard", "zero aura", "main character energy"]
    slang_answers = [
        "I know, that came out of nowhere.", "Yeah, the situation is getting ridiculous.",
        "Fair reaction. I would pause there too.", "That is internet-language chaos, but I am following.",
        "You sound unconvinced. What part lost you?", "Honestly? That has meme energy.",
    ]
    for prompt in slang_starts:
        for answer in rng.sample(slang_answers, 4):
            add(row((prompt, answer)), answer_cap=4)

    typo_pairs = [
        ("hu are u", "I am ChudGPT-Public, a small experimental conversational language model."),
        ("wats chud gpt public", "ChudGPT-Public is the independently trained public version of ChudGPT, built as a small decoder-only transformer."),
        ("explane that pls", "Sure. Tell me which part you want explained, and I will make it simpler."),
        ("wht do u mean", "I mean the point from my previous message. Which part sounded unclear?"),
        ("can u say it agin", "Yes. I can restate it more clearly."),
    ]
    for prompt, answer in typo_pairs:
        add(row((prompt, answer)))

    # Dense contextual micro-dialogues make short turns meaningful without
    # memorizing the held-out phrases. Each answer carries a different subject
    # or value, so the model cannot win by emitting one universal short reply.
    nouns = ["lantern", "spaceship", "keyboard", "helmet", "poster", "camera", "robot", "garden", "puzzle", "playlist"]
    moods = ["calm", "chaotic", "bright", "strange", "funny", "dramatic", "cozy", "serious"]
    for index in range(1_600):
        noun, mood = rng.choice(nouns), rng.choice(moods)
        value = 200 + index
        kind = index % 8
        if kind == 0:
            add(row((f"The {noun} should feel {mood}.", f"Okay, I will keep the {noun} {mood}."), ("why", f"Because {mood} matches the direction you gave for the {noun}.")))
        elif kind == 1:
            add(row((f"I scored {value} points.", f"Nice, {value} points is a specific score."), ("lol", "Yeah, that score sounds like it came from a ridiculous round.")))
        elif kind == 2:
            add(row((f"Pick between the {noun} and the robot.", f"I pick the {noun}."), ("nah", f"Fair enough; we can use the robot instead of the {noun}.")))
        elif kind == 3:
            add(row((f"The secret number is {value}.", f"Got it, the secret number is {value}."), ("what", f"You said the secret number was {value}.")))
        elif kind == 4:
            add(row((f"Tell me a {mood} idea.", f"Imagine a {noun} that only works when nobody is looking."), ("again", f"Another {mood} idea: a tiny robot tries to return a lost {noun} across a huge city.")))
        elif kind == 5:
            add(row((f"We were discussing the {noun}.", f"Right, our topic is the {noun}."), ("tell me more", f"A useful next step is deciding what the {noun} does, who uses it, and why it matters.")))
        elif kind == 6:
            add(row((f"Choose a label for option {value}.", f"I would label option {value} as {mood.title()} {noun.title()}."), ("huh", f"I chose a playful label for option {value}; I can make it plainer if you want.")))
        else:
            add(row((f"I do not like the {mood} version.", f"Understood; we will drop the {mood} version."), ("yes", f"All right, the {mood} version stays removed.")))

    identity_prompts = ["What kind of model are you?", "Tell me who you are.", "Which assistant is this?", "Describe this AI.", "What does Public mean in your name?"]
    identity_answers = [
        "I am ChudGPT-Public, a small experimental decoder-only transformer language model.",
        "This is ChudGPT-Public, the independently trained public ChudGPT model. I generate text from learned patterns and the current conversation.",
        "Public means I am the separately packaged public ChudGPT model, not the Pro, Plus, Buggy, or Mega profile.",
    ]
    for prompt in identity_prompts:
        for answer in identity_answers:
            add(row((prompt, answer)), answer_cap=4)

    # Remove exact held-out user strings if a source happened to contain them.
    records = [candidate for candidate in records if not any(message["role"] == "user" and message["content"].strip().lower() in HELDOUT for message in candidate["messages"])]
    # Fill the remaining broad-knowledge budget with clean source records under
    # a still-low exact-answer cap. The first pass used a cap of two; four keeps
    # any one response far below the old eight-copy alignment maximum.
    rng.shuffle(source_rows)
    for candidate in source_rows:
        add(candidate, answer_cap=4)
        if len(records) >= TARGET:
            break
    rng.shuffle(records)
    if len(records) < TARGET:
        raise RuntimeError(f"Only {len(records)} clean records were built; need {TARGET}")
    return records[:TARGET]


def main() -> None:
    records = build()
    OUTPUT.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
    print(f"Wrote {len(records):,} clean Public v9 conversations to {OUTPUT}")


if __name__ == "__main__":
    main()
