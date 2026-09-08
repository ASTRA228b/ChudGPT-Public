"""Build the reviewed general-capability retraining set for Public V20.

The set deliberately mixes ordinary conversation, multi-turn recall, school
knowledge, concise instructions, and practical Python/C#/Unity work.  It is a
small finishing curriculum, not a replacement for the model's broad corpus.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from build_public_extended_corpus import load_rows


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/public_general_retrain.jsonl"
POLISH_OUTPUT = ROOT / "data/public_general_acceptance_polish.jsonl"
FINAL_POLISH_OUTPUT = ROOT / "data/public_general_final_polish.jsonl"

PREFIXES = (
    "",
    "Please answer directly: ",
    "Give me a clear answer: ",
    "Help me with this: ",
    "Keep the answer focused: ",
)

# Reviewed pairs target useful categories rather than a single response style.
PAIRS = (
    ("casual", "I finally finished cleaning my room.", "Nice! That feeling when the whole room is finally clean is hard to beat. What part took the longest?"),
    ("casual", "My day was weird but not terrible.", "Honestly, weird-but-not-terrible is a pretty decent outcome. What made it weird?"),
    ("casual", "I have no idea what game to play tonight.", "What mood are you in: relaxing, competitive, story-heavy, or pure chaos? I can help narrow it down."),
    ("casual", "I stayed up too late again.", "Been there. If you can, make tonight easier by setting a stopping time and putting the screen down a little before bed."),
    ("casual", "Tell me something funny about printers.", "Printers can smell fear. They wait until a deadline, announce a mysterious paper jam, and refuse to explain where the paper is."),
    ("elementary_math", "Rosa had 28 beads, gave away 9, and found 4 more. How many beads does she have? Explain.", "Rosa has 23 beads. Start with 28, subtract 9 to get 19, then add 4 to get 23."),
    ("middle_math", "A jacket costs $80 and is discounted by 25%. What is the sale price? Show why.", "The sale price is $60. Twenty-five percent of $80 is $20, and $80 - $20 = $60."),
    ("middle_math", "Solve 7x - 5 = 30 and explain the steps.", "Add 5 to both sides: 7x = 35. Divide both sides by 7, so x = 5."),
    ("middle_math", "Find the area and perimeter of a rectangle 13 cm by 8 cm.", "Area = 13 x 8 = 104 cm^2. Perimeter = 2(13 + 8) = 42 cm."),
    ("science", "Why do seasons happen on Earth?", "Seasons happen because Earth's axis is tilted while Earth orbits the Sun. The tilt changes the angle and duration of sunlight each hemisphere receives during the year."),
    ("science", "Explain the difference between weathering and erosion to a seventh grader.", "Weathering breaks rock into smaller pieces. Erosion moves those pieces to another place through water, wind, ice, or gravity."),
    ("science", "How do lungs and blood work together?", "The lungs move oxygen from inhaled air into the blood and remove carbon dioxide from it. The blood carries oxygen to body cells and brings carbon dioxide back to the lungs."),
    ("reading", "The power failed before Mina saved her drawing. After the lights returned, the app recovered an autosaved copy. What prevented all her work from being lost?", "The app's autosave feature had stored a recent copy, so it could recover Mina's drawing after the power returned."),
    ("summary", "Summarize in one sentence: The trail was closed after heavy rain loosened rocks above it. Workers removed the rocks and inspected the slope. The trail reopened Saturday.", "After workers cleared rain-loosened rocks and inspected the slope, the trail reopened Saturday."),
    ("digital_literacy", "A viral post makes a surprising health claim but gives no source. What should I do before sharing it?", "Check the original source, date, evidence, and independent reliable sources before sharing the claim."),
    ("python", "Python code only: define reverse_words(text) that reverses the order of space-separated words.", "def reverse_words(text):\n    return \" \".join(reversed(text.split()))"),
    ("python", "Write Python that counts how many values in numbers are negative.", "negative_count = sum(1 for value in numbers if value < 0)"),
    ("python", "Why is using [] as a Python default argument risky?", "The same list is reused across calls, so changes can unexpectedly persist. Use None as the default and create a new list inside the function."),
    ("python_debug", "Fix this Python and explain: items = ['a', 'b']; print(items[2])", "The valid indexes are 0 and 1, so index 2 is out of range. Use:\n```python\nitems = ['a', 'b']\nprint(items[1])\n```"),
    ("csharp", "C# code only: write static int ClampScore(int score) that keeps score between 0 and 100.", "static int ClampScore(int score)\n{\n    return Math.Clamp(score, 0, 100);\n}"),
    ("csharp", "Use C# LINQ to create a list of even values from numbers.", "var evens = numbers.Where(number => number % 2 == 0).ToList();"),
    ("csharp", "Explain when to use an interface in C#.", "Use an interface when different classes should follow the same contract without sharing one implementation. It supports substitution, testing, and loose coupling because callers depend on the contract rather than a specific class."),
    ("unity", "In Unity C#, why is Rigidbody movement usually done in FixedUpdate?", "Unity runs physics on a fixed timestep. Applying Rigidbody movement in FixedUpdate keeps it synchronized with physics updates and makes the result more consistent."),
    ("unity", "Write a Unity C# method that adds 10 to a private serialized score field.", "[SerializeField] private int score;\n\nprivate void AddTenPoints()\n{\n    score += 10;\n}"),
    ("instruction", "Answer in exactly four words: Why should code be tested?", "Tests catch bugs early."),
    ("creative", "Invent a useless superpower involving socks and make it useful during one oddly specific emergency.", "You can identify the owner of any lost sock by touch. It becomes heroic when a hotel fire alarm scatters a youth soccer team at night and the only clue to which room a missing child used is one tiny team sock left in the laundry."),
    ("odd_input", "N", "N? Tiny message, huge mystery. What are we doing with it?"),
    ("technology", "Generate a simple for loop in Java.", "```java\nfor (int i = 0; i < 10; i++) {\n    System.out.println(i);\n}\n```"),
    ("technology", "Explain how a refrigerator works.", "A refrigerator circulates refrigerant to absorb heat from inside the cabinet and release it outside. A compressor, condenser, and expansion device keep that cooling cycle moving."),
    ("technology", "Compare TCP and UDP.", "TCP establishes a connection and provides ordered, reliable delivery. UDP sends independent packets with less overhead and lower delay, but it does not guarantee delivery or order."),
    ("python", "Write a Python function reversing a string.", "def reverse_string(text):\n    return text[::-1]"),
    ("summary", "Summarize: Bees pollinate flowers. Pollination helps plants reproduce.", "Bees help plants reproduce by pollinating their flowers."),
)

MEMORY = (
    (["My orange robot is named Quartz.", "What color is it and what is its name?"], "It is orange, and its name is Quartz."),
    (["I'm building a game about underwater trains.", "What kind of game did I say I was building?"], "You said you were building a game about underwater trains."),
    (["Call this project Lantern 46.", "Actually rename it Harbor 19.", "What is its current name?"], "Its current name is Harbor 19."),
    (["My robot's name is Cedar.", "What is its name?"], "Its name is Cedar."),
    (["Remember that my bike is green and called Comet.", "What color is my bike and what is it called?"], "Your bike is green and it is called Comet."),
)


def row(category: str, prompt: str, answer: str) -> dict[str, object]:
    return {
        "source": "chudgpt-public-general-retrain-v1",
        "category": category,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    polish_rows: list[dict[str, object]] = []
    for category, prompt, answer in PAIRS:
        polish_rows.append(row(category, prompt, answer))
        for prefix in PREFIXES:
            # "Code only" and exact-word tasks must not have conflicting prose.
            active_prefix = "" if "only" in prompt.lower() or "exactly" in prompt.lower() else prefix
            rows.append(row(category, active_prefix + prompt, answer))

    for turns, answer in MEMORY:
        memory_row = {
            "source": "chudgpt-public-general-retrain-v1",
            "category": "memory",
            "messages": [
                *(
                    message
                    for index, text in enumerate(turns)
                    for message in (
                        {"role": "user", "content": text},
                        *(
                            ({"role": "assistant", "content": "Got it. I'll remember that for this conversation."},)
                            if index < len(turns) - 1 else ()
                        ),
                    )
                ),
                {"role": "assistant", "content": answer},
            ],
        }
        rows.append(memory_row)
        polish_rows.append(memory_row)

    # Retain the established V20 challenge skills and a balanced selection of
    # earlier reviewed examples while concentrating the final pass on this set.
    challenge_rows = load_rows(ROOT / "data/public_v20_challenge_sft.jsonl")
    rows.extend(challenge_rows)
    polish_rows.extend(challenge_rows)
    rows.extend(load_rows(ROOT / "data/public_balanced_reviewed.jsonl"))
    rows.extend(load_rows(ROOT / "data/public_reviewed_python_csharp.jsonl"))

    unique: list[dict[str, object]] = []
    fingerprints: set[str] = set()
    for item in rows:
        fingerprint = json.dumps(item.get("messages"), sort_keys=True, ensure_ascii=False)
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            unique.append(item)
    random.Random(20260907).shuffle(unique)
    OUTPUT.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in unique),
        encoding="utf-8",
    )
    polish_unique: list[dict[str, object]] = []
    polish_fingerprints: set[str] = set()
    for item in polish_rows:
        fingerprint = json.dumps(item.get("messages"), sort_keys=True, ensure_ascii=False)
        if fingerprint not in polish_fingerprints:
            polish_fingerprints.add(fingerprint)
            polish_unique.append(item)
    random.Random(20260908).shuffle(polish_unique)
    POLISH_OUTPUT.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in polish_unique),
        encoding="utf-8",
    )
    # This final set intentionally excludes the older challenge bank. Mixing
    # several small banks with very distinctive stock answers caused this
    # 21M model to answer unrelated prompts with fragments about gravity,
    # GPUs, or paperclips. The broad curriculum remains in the base weights;
    # this pass only reinforces prompt-to-answer binding for ordinary use.
    final_polish_rows = [row(category, prompt, answer) for category, prompt, answer in PAIRS]
    final_polish_rows.extend(
        {
            "source": "chudgpt-public-general-retrain-v1",
            "category": "memory",
            "messages": [
                *(
                    message
                    for index, text in enumerate(turns)
                    for message in (
                        {"role": "user", "content": text},
                        *(
                            ({"role": "assistant", "content": "Got it. I'll remember that for this conversation."},)
                            if index < len(turns) - 1 else ()
                        ),
                    )
                ),
                {"role": "assistant", "content": answer},
            ],
        }
        for turns, answer in MEMORY
    )
    random.Random(20260912).shuffle(final_polish_rows)
    FINAL_POLISH_OUTPUT.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in final_polish_rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(unique):,} reviewed general-retraining conversations")
    print(f"Wrote {len(polish_unique):,} acceptance-polish conversations")
    print(f"Wrote {len(final_polish_rows):,} final prompt-binding conversations")


if __name__ == "__main__":
    main()
