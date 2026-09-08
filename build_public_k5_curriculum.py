"""Create a deterministic K-5 fundamentals curriculum for Public V20.

These are training examples, not runtime answer tables.  The curriculum uses
many values, wordings, subjects, and multi-turn contexts so the small model can
learn elementary skills without anchoring to a handful of repeated sentences.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/public_k5_curriculum.jsonl"


def main() -> None:
    rng = random.Random(20260907)
    rows: list[dict[str, object]] = []

    def add(grade: int, subject: str, user: str, assistant: str) -> None:
        rows.append({
            "source": "chudgpt-public-k5-v1",
            "grade": grade,
            "category": subject,
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        })

    def add_dialogue(grade: int, subject: str, *turns: str) -> None:
        rows.append({
            "source": "chudgpt-public-k5-v1",
            "grade": grade,
            "category": subject,
            "messages": [
                {"role": "user" if index % 2 == 0 else "assistant", "content": turn}
                for index, turn in enumerate(turns)
            ],
        })

    # Kindergarten and grade-one counting, comparison, addition, and subtraction.
    for left in range(0, 31):
        for right in range(0, 21):
            total = left + right
            grade = 0 if total <= 10 else 1
            wording = rng.choice([
                f"What is {left} + {right}?",
                f"Add {left} and {right}.",
                f"Find the sum of {left} and {right}.",
            ])
            add(grade, "arithmetic", wording, f"{left} + {right} = {total}.")
    for whole in range(1, 51):
        for removed in range(0, min(whole, 15) + 1):
            difference = whole - removed
            wording = rng.choice([
                f"What is {whole} - {removed}?",
                f"Subtract {removed} from {whole}.",
                f"Find the difference between {whole} and {removed}.",
            ])
            add(1 if whole <= 20 else 2, "arithmetic", wording, f"{whole} - {removed} = {difference}.")

    # Grades two through five: multiplication, division, place value, fractions,
    # decimals, and multi-step arithmetic with brief reasoning.
    for left in range(1, 13):
        for right in range(1, 13):
            product = left * right
            add(3, "multiplication", f"What is {left} times {right}?",
                f"{left} × {right} = {product}.")
            add(3, "division", f"What is {product} divided by {left}?",
                f"{product} ÷ {left} = {right}, because {left} × {right} = {product}.")
    for number in range(101, 1000, 7):
        hundreds = number // 100
        tens = (number // 10) % 10
        ones = number % 10
        add(2, "place_value", f"Write {number} in expanded form.",
            f"{number} = {hundreds * 100} + {tens * 10} + {ones}.")
        place = rng.choice(("hundreds", "tens", "ones"))
        value = {"hundreds": hundreds, "tens": tens, "ones": ones}[place]
        add(2, "place_value", f"What digit is in the {place} place in {number}?", str(value))
    for denominator in range(2, 13):
        for numerator in range(1, denominator):
            add(3, "fractions", f"In the fraction {numerator}/{denominator}, what is the denominator?",
                f"The denominator is {denominator}. It tells how many equal parts make the whole.")
            complement = denominator - numerator
            add(4, "fractions", f"What must be added to {numerator}/{denominator} to make one whole?",
                f"Add {complement}/{denominator}, because {numerator}/{denominator} + {complement}/{denominator} = {denominator}/{denominator} = 1.")
    for whole in range(1, 31):
        for tenths in (1, 2, 4, 5, 7, 8):
            value = whole + tenths / 10
            add(4, "decimals", f"Write {value:.1f} as a mixed number with tenths.",
                f"{value:.1f} is {whole} and {tenths}/10.")
    for a in range(2, 22, 2):
        b = a + 3
        c = (a % 5) + 2
        result = (a + b) * c
        add(5, "order_of_operations", f"Calculate ({a} + {b}) × {c}. Show the two steps.",
            f"First, {a} + {b} = {a + b}. Then {a + b} × {c} = {result}.")

    names = ["Ava", "Ben", "Cora", "Diego", "Eli", "Fatima", "Grace", "Hugo", "Iris", "Jamal"]
    items = ["apples", "marbles", "stickers", "books", "shells", "crayons", "cards", "toy cars"]
    for index in range(500):
        name = names[index % len(names)]
        item = items[(index * 3) % len(items)]
        start = 5 + index % 46
        change = 1 + (index * 7) % min(20, start)
        if index % 2:
            result = start + change
            prompt = f"{name} has {start} {item} and gets {change} more. How many {item} does {name} have now?"
            answer = f"{name} has {start + change} {item}: {start} + {change} = {result}."
        else:
            result = start - change
            prompt = f"{name} has {start} {item} and gives away {change}. How many are left?"
            answer = f"{result} are left: {start} - {change} = {result}."
        add(1 if start <= 20 else 2, "word_problems", prompt, answer)
    for index in range(240):
        groups = 2 + index % 11
        per_group = 2 + (index * 5) % 10
        total = groups * per_group
        item = items[index % len(items)]
        if index % 2:
            prompt = f"There are {groups} boxes with {per_group} {item} in each box. How many {item} are there altogether?"
            answer = f"There are {total} {item}: {groups} groups × {per_group} in each group = {total}."
        else:
            prompt = f"{total} {item} are shared equally among {groups} students. How many does each student get?"
            answer = f"Each student gets {per_group}: {total} ÷ {groups} = {per_group}."
        add(3, "word_problems", prompt, answer)

    # Reading, vocabulary, grammar, and concise comprehension.
    passages = [
        ("Maya planted three bean seeds. She watered them every morning. A week later, two green shoots appeared.",
         "What did Maya plant?", "Maya planted three bean seeds."),
        ("The class put ice in two cups. One cup sat in sunlight and the other sat in shade. The ice in sunlight melted first.",
         "Which ice melted first?", "The ice in the cup that sat in sunlight melted first."),
        ("Leo packed a map, water, and a red jacket before the hike. Clouds gathered, so he wore the jacket.",
         "Why did Leo wear his jacket?", "Leo wore his jacket because clouds gathered and the weather might turn wet or cold."),
        ("Nora's library book was due Friday. She finished it Thursday night and returned it before school on Friday.",
         "Did Nora return the book late? Explain.", "No. She returned it Friday morning, which was still the due date."),
        ("A fox saw its reflection in a pond. It barked, and the image seemed to bark back. Then the ripples broke the image apart.",
         "What caused the reflection to break apart?", "Ripples in the pond caused the reflection to break apart."),
        ("The soccer game was canceled because lightning was seen near the field. The team practiced indoors instead.",
         "What did the team do after the game was canceled?", "The team practiced indoors."),
    ]
    for cycle in range(80):
        passage, question, answer = passages[cycle % len(passages)]
        add(2 + cycle % 4, "reading_comprehension", f"Read this passage: {passage}\n\n{question}", answer)
        add(2 + cycle % 4, "summarization", f"Summarize in one sentence: {passage}",
            rng.choice([
                passage.split(". ")[0] + ".",
                " ".join(passage.split()[:12]).rstrip(".,") + ".",
            ]))
    synonyms = [
        ("happy", "glad"), ("large", "big"), ("tiny", "small"), ("quick", "fast"),
        ("begin", "start"), ("silent", "quiet"), ("angry", "mad"), ("smart", "clever"),
        ("finish", "complete"), ("simple", "easy"), ("brave", "courageous"), ("choose", "select"),
    ]
    antonyms = [
        ("hot", "cold"), ("early", "late"), ("empty", "full"), ("smooth", "rough"),
        ("bright", "dark"), ("inside", "outside"), ("above", "below"), ("ancient", "modern"),
    ]
    for cycle in range(20):
        for word, synonym in synonyms:
            add(2, "vocabulary", f"Give one synonym for {word}.", synonym)
        for word, antonym in antonyms:
            add(2, "vocabulary", f"Give one antonym for {word}.", antonym)
    for singular, plural in [("cat", "cats"), ("box", "boxes"), ("baby", "babies"),
                             ("leaf", "leaves"), ("child", "children"), ("mouse", "mice")]:
        for wording in (f"What is the plural of {singular}?", f"Make the word '{singular}' plural."):
            add(2, "grammar", wording, plural)
    for sentence, corrected in [
        ("the dog ran home", "The dog ran home."),
        ("we saw a rainbow", "We saw a rainbow."),
        ("my friend likes pizza", "My friend likes pizza."),
        ("where is the pencil", "Where is the pencil?"),
        ("what time is lunch", "What time is lunch?"),
    ]:
        for _ in range(16):
            add(1, "grammar", f"Fix the capitalization and punctuation: {sentence}", corrected)

    # Elementary science and social studies facts, with multiple question forms.
    facts = [
        (1, "Why do plants need sunlight?", "Plants use sunlight to make food through photosynthesis."),
        (1, "What do roots do for a plant?", "Roots hold the plant in place and take in water and minerals from the soil."),
        (1, "What happens when water freezes?", "Liquid water changes into solid ice."),
        (2, "Why do shadows form?", "A shadow forms when an object blocks light."),
        (2, "What is a habitat?", "A habitat is the place where an organism lives and finds what it needs."),
        (2, "Why do we have day and night?", "Earth rotates, so different parts face toward or away from the Sun."),
        (3, "What is evaporation?", "Evaporation is the change from liquid water into water vapor."),
        (3, "How are weather and climate different?", "Weather is what the atmosphere is like over a short time; climate is the usual pattern over many years."),
        (3, "What is a food chain?", "A food chain shows how energy moves from one organism to another through eating."),
        (4, "Why does the Moon seem to change shape?", "As the Moon orbits Earth, we see different amounts of its sunlit half."),
        (4, "What is erosion?", "Erosion is the movement of rock and soil by water, wind, ice, or gravity."),
        (4, "What is an electrical circuit?", "A circuit is a complete path through which electric current can flow."),
        (5, "Why do objects fall toward Earth?", "Earth's gravity accelerates objects toward its center."),
        (5, "What is the difference between a physical and chemical change?", "A physical change alters form or state, while a chemical change makes one or more new substances."),
        (5, "Why are decomposers important?", "Decomposers break down dead material and return nutrients to ecosystems."),
        (3, "What is a community?", "A community is a group of people who live, work, or share something together."),
        (3, "Why do communities have rules?", "Rules help people stay safe, resolve conflicts, and understand expected behavior."),
        (4, "What is a primary source?", "A primary source is an original record or object from the time being studied."),
    ]
    for repeat in range(12):
        for grade, question, answer in facts:
            add(grade, "elementary_knowledge", question, answer)

    # Following constraints and conversational memory are necessary foundations
    # for both schoolwork and longer Discord conversations.
    colors = ["red", "blue", "green", "purple", "orange", "yellow"]
    animals = ["otter", "tiger", "rabbit", "panda", "falcon", "turtle"]
    for index in range(180):
        color = colors[index % len(colors)]
        animal = animals[(index * 5) % len(animals)]
        add(2, "instruction_following", f"Reply with exactly two words: {color} {animal}", f"{color} {animal}")
        add(3, "instruction_following", f"Put these words in alphabetical order: {animal}, apple, {color}.",
            ", ".join(sorted((animal, "apple", color))))
    for index in range(120):
        name = f"Project {chr(65 + index % 26)}{index}"
        topic = ["space", "oceans", "robots", "gardens", "weather"][index % 5]
        add_dialogue(2 + index % 4, "memory",
                     f"My project is called {name}, and it is about {topic}.",
                     f"Got it. {name} is your project about {topic}.",
                     "What is my project called, and what is it about?",
                     f"Your project is called {name}, and it is about {topic}.")

    rng.shuffle(rows)
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows):,} K-5 curriculum examples to {OUTPUT}")


if __name__ == "__main__":
    main()
