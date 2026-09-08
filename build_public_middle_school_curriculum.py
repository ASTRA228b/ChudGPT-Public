"""Build grades 6-8 fundamentals and a combined K-8 curriculum."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MIDDLE_OUTPUT = ROOT / "data/public_middle_school_curriculum.jsonl"
COMBINED_OUTPUT = ROOT / "data/public_k8_curriculum.jsonl"
K5_INPUT = ROOT / "data/public_k5_curriculum.jsonl"


def main() -> None:
    rng = random.Random(20260908)
    rows: list[dict[str, object]] = []

    def add(grade: int, subject: str, prompt: str, answer: str) -> None:
        rows.append({
            "source": "chudgpt-public-middle-school-v1",
            "grade": grade,
            "category": subject,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
        })

    # Number sense, ratios, percentages, algebra, geometry, and statistics.
    for left in range(-25, 26):
        for right in range(-12, 13, 3):
            add(6, "integers", f"Calculate {left} + ({right}).",
                f"{left} + ({right}) = {left + right}.")
            add(6, "integers", f"Calculate {left} - ({right}).",
                f"{left} - ({right}) = {left - right}.")
    for base in range(10, 101, 5):
        for percent in (5, 10, 20, 25, 50, 75):
            result = base * percent / 100
            shown = str(int(result)) if result.is_integer() else f"{result:g}"
            add(6, "percent", f"What is {percent}% of {base}? Show the calculation.",
                f"{percent}% of {base} is {percent / 100:g} × {base} = {shown}.")
    for first in range(2, 13):
        for second in range(2, 9):
            scale = 2 + (first + second) % 5
            add(6, "ratios", f"A recipe uses {first} cups of water for {second} cups of rice. At the same ratio, how much water is needed for {second * scale} cups of rice?",
                f"The rice amount was multiplied by {scale}, so multiply the water by {scale}: {first} × {scale} = {first * scale} cups.")
    for index in range(420):
        coefficient = 2 + index % 11
        solution = -10 + (index * 7) % 31
        constant = -15 + (index * 5) % 31
        total = coefficient * solution + constant
        add(7, "algebra", f"Solve for x: {coefficient}x + ({constant}) = {total}.",
            f"Subtract {constant} from both sides to get {coefficient}x = {coefficient * solution}. Divide by {coefficient}, so x = {solution}.")
    for index in range(180):
        length = 3 + index % 28
        width = 2 + (index * 5) % 19
        add(6, "geometry", f"A rectangle is {length} cm long and {width} cm wide. Find its area and perimeter.",
            f"Area = {length} × {width} = {length * width} cm². Perimeter = 2({length} + {width}) = {2 * (length + width)} cm.")
    for radius in range(1, 21):
        add(7, "geometry", f"Using π ≈ 3.14, find the circumference of a circle with radius {radius} cm.",
            f"C = 2πr ≈ 2 × 3.14 × {radius} = {2 * 3.14 * radius:.2f} cm.")
    for index in range(160):
        values = [2 + (index * factor) % 19 for factor in (3, 5, 7, 11, 13)]
        mean = sum(values) / len(values)
        add(6, "statistics", f"Find the mean of {', '.join(map(str, values))}.",
            f"Their sum is {sum(values)}. Divide by {len(values)} to get a mean of {mean:g}.")
    for favorable in range(1, 10):
        for total in range(favorable + 1, 13):
            divisor = math.gcd(favorable, total)
            add(7, "probability", f"A bag has {total} equally likely tokens, and {favorable} are blue. What is the probability of drawing blue?",
                f"The probability is {favorable}/{total}, which simplifies to {favorable // divisor}/{total // divisor}.")

    science = [
        (6, "cells", "What is the main job of the cell membrane?", "The cell membrane controls what enters and leaves the cell and helps protect the cell."),
        (6, "cells", "How are plant and animal cells different?", "Plant cells have cell walls and chloroplasts; animal cells do not. Both have membranes, cytoplasm, and genetic material."),
        (6, "ecology", "What is an ecosystem?", "An ecosystem includes a community of organisms and the nonliving environment they interact with."),
        (7, "genetics", "What is a gene?", "A gene is a section of DNA that carries information affecting a trait or biological function."),
        (7, "body_systems", "How do the respiratory and circulatory systems work together?", "The respiratory system exchanges oxygen and carbon dioxide, while the circulatory system transports those gases through the body."),
        (8, "evolution", "What is natural selection?", "Natural selection is the process in which inherited traits that improve survival or reproduction become more common over generations."),
        (6, "earth_science", "What causes the seasons?", "Earth's axis is tilted. As Earth orbits the Sun, each hemisphere receives different angles and lengths of sunlight during the year."),
        (6, "earth_science", "How are weathering and erosion different?", "Weathering breaks rock down; erosion moves the broken material to another place."),
        (7, "plate_tectonics", "Why do many earthquakes happen near plate boundaries?", "Tectonic plates push, pull, or slide past one another. Built-up stress can release suddenly as an earthquake."),
        (8, "astronomy", "Why does gravity keep planets in orbit?", "A planet's forward motion and the star's inward gravitational acceleration combine to produce a curved orbital path."),
        (6, "matter", "How are atoms, elements, and compounds related?", "Atoms are basic units of matter. An element contains one kind of atom, while a compound contains atoms of different elements chemically joined."),
        (7, "chemistry", "What is the difference between a mixture and a compound?", "A mixture combines substances physically, so they keep their properties; a compound joins elements chemically in fixed proportions."),
        (7, "forces", "What is the difference between speed and velocity?", "Speed tells how fast something moves. Velocity includes both speed and direction."),
        (8, "forces", "State Newton's third law in simple language.", "When one object exerts an interaction force on another, the second exerts an equal force in the opposite direction on the first."),
        (8, "energy", "Why is energy conserved?", "Energy can transfer between objects or change forms, but the total amount in a closed system remains constant."),
        (7, "waves", "How are frequency and wavelength related for a wave traveling at constant speed?", "At a constant wave speed, higher frequency means shorter wavelength, and lower frequency means longer wavelength."),
    ]
    for repeat in range(20):
        for grade, topic, question, answer in science:
            add(grade, f"science_{topic}", question, answer)

    passages = [
        ("The town replaced several parking spaces with a small public garden. Some shop owners worried about fewer customers, but foot traffic rose after benches and shade trees were added.",
         "What evidence challenges the shop owners' worry?", "Foot traffic increased after the garden, benches, and shade trees were added."),
        ("Rina tested three paper-airplane designs. She used the same paper and threw each plane from the same line. The plane with wider wings traveled farthest in four of five trials.",
         "Why did Rina use the same paper and starting line?", "Keeping those conditions the same made the wing design the main variable being tested."),
        ("The article says school lunches should include more fresh fruit because students need nutritious choices. It also reports that fruit was the most discarded food last month.",
         "What tension appears in the article?", "The article supports serving more fruit, but its own evidence shows students often discard it."),
        ("After the river flooded, volunteers first delivered drinking water. Weeks later, they helped repair homes and restore the damaged trail.",
         "How did the volunteers' work change over time?", "It shifted from immediate emergency aid to longer-term rebuilding and restoration."),
        ("Kai assumed the old laptop was useless. After installing a lightweight operating system, he used it to write, browse reference pages, and learn programming.",
         "What lesson does the passage suggest?", "Old hardware can remain useful when software and expectations fit its limits."),
    ]
    for cycle in range(100):
        passage, question, answer = passages[cycle % len(passages)]
        add(6 + cycle % 3, "reading_inference", f"Read the passage and answer: {passage}\n\n{question}", answer)
        add(6 + cycle % 3, "summary", f"Summarize this passage in one sentence: {passage}",
            rng.choice([
                answer,
                "The passage explains " + answer[0].lower() + answer[1:],
            ]))
    arguments = [
        ("Schools should plant shade trees because they cool outdoor spaces and provide habitat.", "Schools should plant shade trees.", "They cool outdoor spaces and provide habitat."),
        ("The library should stay open later because students need a quiet place to study after activities.", "The library should stay open later.", "Students need a quiet study place after activities."),
        ("Reusable bottles reduce disposable plastic waste, so students should bring them when possible.", "Students should bring reusable bottles when possible.", "They reduce disposable plastic waste."),
    ]
    for cycle in range(90):
        text, claim, evidence = arguments[cycle % len(arguments)]
        add(7, "argument", f"Identify the claim and supporting reason: {text}", f"Claim: {claim} Reason: {evidence}")

    # Digital literacy and beginning programming, phrased as explanations and code.
    digital = [
        ("Why should you verify a surprising claim online?", "A claim can be wrong, outdated, or misleading. Check it against reliable independent sources and the original evidence."),
        ("What makes a password stronger?", "Use a long, unique password or passphrase for each account and store it in a reputable password manager."),
        ("What is phishing?", "Phishing is an attempt to trick someone into revealing information or installing something harmful by pretending to be trustworthy."),
        ("What is an algorithm?", "An algorithm is a clear sequence of steps for solving a problem or completing a task."),
        ("Why cite a source?", "A citation credits the source and lets readers check where information came from."),
    ]
    for repeat in range(24):
        for question, answer in digital:
            add(6, "digital_literacy", question, answer)
    for index in range(180):
        name = ["score", "level", "health", "coins", "speed", "count"][index % 6]
        amount = 1 + index % 20
        add(6, "programming", f"Write one Python line that adds {amount} to the variable {name}.", f"{name} += {amount}")
    for index in range(120):
        limit = 2 + index % 12
        add(7, "programming", f"Write a Python loop that prints the integers from 0 through {limit}. Code only.",
            f"for number in range({limit + 1}):\n    print(number)")
    for index in range(90):
        value = 2 + index % 50
        add(7, "debugging", f"This Python should print {value * 2}, but it joins text instead: value = '{value}'; print(value + value). Fix it and explain.",
            f"```python\nvalue = {value}\nprint(value + value)\n```\nThe quotes made the value a string. Using an integer makes addition numeric, producing {value * 2}.")

    rng.shuffle(rows)
    MIDDLE_OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    if not K5_INPUT.is_file():
        raise FileNotFoundError(f"Build the K-5 curriculum first: {K5_INPUT}")
    combined = K5_INPUT.read_text(encoding="utf-8") + MIDDLE_OUTPUT.read_text(encoding="utf-8")
    COMBINED_OUTPUT.write_text(combined, encoding="utf-8")
    print(f"Wrote {len(rows):,} grades 6-8 examples to {MIDDLE_OUTPUT}")
    print(f"Combined K-8 curriculum: {sum(1 for line in combined.splitlines() if line.strip()):,} examples")


if __name__ == "__main__":
    main()
