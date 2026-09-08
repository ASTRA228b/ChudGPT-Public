"""Build reviewed multi-turn and everyday-life instruction data for Public V20.

The rows teach conversational state and broadly reusable practical reasoning.
They are training examples, not runtime response rules or fallback text.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/public_dialogue_life_curriculum.jsonl"


def row(category: str, messages: list[tuple[str, str]]) -> dict[str, object]:
    return {
        "source": "chudgpt-public-dialogue-life-v1",
        "category": category,
        "messages": [{"role": role, "content": content} for role, content in messages],
    }


def memory_rows() -> list[dict[str, object]]:
    people = ("Maya", "Theo", "Nora", "Jules", "Iris", "Caleb", "Zoe", "Omar")
    objects = ("bicycle", "notebook", "backpack", "controller", "mug", "jacket", "headphones", "lunchbox")
    colors = ("teal", "silver", "purple", "green", "orange", "navy", "yellow", "red")
    places = ("garage", "desk", "hall closet", "kitchen", "bedroom", "workbench", "car", "school locker")
    rows: list[dict[str, object]] = []
    for index, (person, item, color, place) in enumerate(itertools.islice(itertools.product(people, objects, colors, places), 320)):
        if index % 2 == 0:
            messages = [
                ("user", f"Remember this for later: {person}'s {item} is {color}."),
                ("assistant", f"Got it. {person}'s {item} is {color}."),
                ("user", f"What color is {person}'s {item}?"),
                ("assistant", f"It is {color}."),
            ]
        else:
            messages = [
                ("user", f"I left my {color} {item} in the {place}."),
                ("assistant", f"Okay, your {color} {item} is in the {place}."),
                ("user", "Where did I leave it?"),
                ("assistant", f"You left it in the {place}."),
            ]
        rows.append(row("multi_turn_memory", messages))

    old_names = ("Comet", "Maple", "Northstar", "Pebble", "Signal", "Willow", "Copper", "Juniper")
    new_names = ("Beacon", "Riverside", "Atlas", "Cinder", "Meadow", "Summit", "Kestrel", "Driftwood")
    for old_name, new_name in itertools.product(old_names, new_names):
        rows.append(row("multi_turn_correction", [
            ("user", f"Call my project {old_name}."),
            ("assistant", f"All right, the project is called {old_name}."),
            ("user", f"Change the project name to {new_name}."),
            ("assistant", f"Updated. Its current name is {new_name}."),
            ("user", "What is its current name?"),
            ("assistant", f"Its current name is {new_name}."),
        ]))
    return rows


PRACTICAL = (
    ("My smoke alarm chirps once every minute. What should I check?", "A periodic chirp usually means the battery is low. Replace the battery, check the alarm's date, and replace the unit if it is past its rated life."),
    ("How can I make a strong password?", "Use a long, unique passphrase for every account and store it in a password manager. Turn on multifactor authentication when it is available."),
    ("I received an unexpected login link by text. What should I do?", "Do not open the link. Go to the service through its official app or typed address, check recent activity, and report the message as phishing."),
    ("How do I avoid mixing dark and light laundry?", "Sort light and dark items before washing, check care labels, and wash new deeply colored clothes separately the first few times."),
    ("Why should raw chicken use a separate cutting board?", "Raw chicken can carry harmful bacteria. Keeping it separate prevents those bacteria from reaching ready-to-eat food, utensils, and surfaces."),
    ("What should I do before installing a program from the web?", "Confirm that the site is the developer's official source, check the publisher and file reputation, scan the download, and avoid bundled offers you did not request."),
    ("How can I start saving money from a weekly allowance?", "Choose a small amount or percentage to save first, put it aside immediately, and track what remains for spending."),
    ("My friend seems upset but says they are fine. What can I say?", "Try a low-pressure response such as, 'I'm here if you want to talk.' Give them space while staying kind and available."),
    ("How do I plan homework when several assignments are due?", "List each task and deadline, estimate the time needed, start with urgent or difficult work, and use short focused blocks with breaks."),
    ("What is a safe first step for a small kitchen grease fire?", "Turn off the heat if you can do so safely and cover the pan with a metal lid. Never pour water on burning grease; leave and call emergency services if it grows."),
    ("Why should I back up files in more than one place?", "A second copy protects you if the computer, drive, account, or original location fails. Important files are safer with both a local and an off-site copy."),
    ("How can I tell whether an online claim is trustworthy?", "Look for the original source, evidence, date, author expertise, and agreement from independent reliable sources before accepting or sharing it."),
    ("My game keeps stuttering. What basic checks should I try?", "Check CPU, GPU, memory, temperature, and disk use; close unnecessary background apps; update drivers; and lower one demanding graphics setting at a time."),
    ("How do I give useful feedback on someone's project?", "Name something specific that works, describe one concrete issue and its effect, and suggest a practical next step without attacking the person."),
    ("Why does a room feel cooler with a fan even if the temperature stays similar?", "Moving air speeds evaporation from your skin and carries away warmed air near your body, so you feel cooler even when room temperature changes little."),
    ("What should I include when reporting a software bug?", "Include the expected result, actual result, exact reproduction steps, version and environment, frequency, and relevant logs or screenshots without private data."),
)


def practical_rows() -> list[dict[str, object]]:
    lead_ins = ("", "Please explain clearly: ", "Give me practical advice. ", "I'm new to this. ")
    rows = []
    for prompt, answer in PRACTICAL:
        for lead_in in lead_ins:
            rows.append(row("everyday_practical", [("user", lead_in + prompt), ("assistant", answer)]))
    return rows


def main() -> None:
    rows = memory_rows() + practical_rows()
    random.Random(20260913).shuffle(rows)
    OUTPUT.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
    print(f"Wrote {len(rows):,} reviewed dialogue and everyday-life conversations")


if __name__ == "__main__":
    main()
