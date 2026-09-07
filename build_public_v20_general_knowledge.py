"""Build a broad V20 curriculum without copying acceptance-test prompts.

The fixed capability suite and ChudLab are evaluation surfaces.  They must never
be read here: this builder teaches reusable facts, explanations, reasoning
patterns, instruction following, and debugging with independent examples.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/public_v20_general_knowledge.jsonl"


def row(user: str, assistant: str, source: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
    }


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    # Short, independently authored reference facts.  Multiple domains matter
    # more than repeating one benchmark-shaped question hundreds of times.
    facts = [
        ("What is photosynthesis?", "Photosynthesis is the process plants, algae, and some bacteria use to turn light energy, carbon dioxide, and water into sugars, releasing oxygen as a by-product."),
        ("Why is the sky usually blue?", "Air molecules scatter short blue wavelengths of sunlight more strongly than longer red wavelengths, so blue light reaches our eyes from across the sky."),
        ("What does DNA do?", "DNA stores biological instructions. Cells read sections called genes to help build proteins and regulate how living things grow and function."),
        ("What is an atom?", "An atom is a basic unit of matter with a nucleus of protons and neutrons surrounded by electrons."),
        ("Why do objects fall near Earth?", "Earth curves spacetime around it, and nearby objects follow paths that accelerate toward the ground when nothing supports them."),
        ("How is speed different from velocity?", "Speed tells how fast something moves, while velocity includes both speed and direction."),
        ("What is electric current?", "Electric current is the rate at which electric charge moves through a material or circuit."),
        ("Why does ice float?", "Frozen water forms an open crystal structure, making ice less dense than liquid water, so buoyancy keeps it afloat."),
        ("What causes seasons on Earth?", "Earth's tilted axis changes the angle and duration of sunlight each hemisphere receives as Earth orbits the Sun."),
        ("What is evolution by natural selection?", "Heritable traits that improve survival or reproduction tend to become more common across generations."),
        ("What does a vaccine teach the body?", "A vaccine trains the immune system to recognize a pathogen or part of one, helping it respond faster during a later exposure."),
        ("What is the water cycle?", "Water evaporates, condenses into clouds, falls as precipitation, and returns through runoff, groundwater, and living things."),
        ("What is a black hole?", "A black hole is a region where spacetime is curved so strongly that beyond its event horizon even light cannot escape."),
        ("Why does the Moon have phases?", "As the Moon orbits Earth, we see changing portions of its sunlit half."),
        ("What is plate tectonics?", "Plate tectonics describes the slow movement of Earth's crustal plates, which helps cause earthquakes, volcanoes, and mountain building."),
        ("What is democracy?", "Democracy is a system in which political authority ultimately comes from the people, commonly through voting and accountable institutions."),
        ("What was the Renaissance?", "The Renaissance was a period of renewed art, scholarship, and scientific inquiry in Europe, especially from the 14th through 17th centuries."),
        ("Why was the printing press important?", "Movable-type printing made books and ideas cheaper to reproduce and much easier to distribute widely."),
        ("What is inflation?", "Inflation is a broad rise in prices over time, which reduces how much a unit of money can buy."),
        ("What does supply and demand describe?", "It describes how the amount sellers offer and buyers want helps shape prices and quantities in a market."),
        ("What is a metaphor?", "A metaphor describes one thing as another to create a comparison, such as calling time a thief."),
        ("What is the main idea of a paragraph?", "The main idea is the central point the paragraph's details explain or support."),
        ("How is a noun different from a verb?", "A noun names a person, place, thing, or idea; a verb expresses an action or state."),
        ("What is an algorithm?", "An algorithm is a finite sequence of steps for solving a problem or completing a task."),
        ("What does an operating system do?", "An operating system manages hardware and resources and provides services that applications use."),
        ("What is an API?", "An API is a defined interface that lets programs request data or actions from another program or service."),
        ("What is a database index?", "A database index is an additional structure that speeds up lookups, usually by trading extra storage and write work for faster reads."),
        ("Why does a GPU contain many processing units?", "A GPU is designed to run many similar calculations in parallel, which suits pixels, vertices, matrices, and machine-learning tensors. A CPU uses fewer, more flexible cores optimized for varied tasks and low-latency control flow."),
        ("What is RAM used for?", "RAM holds data and instructions that running programs need quickly; its contents normally disappear when power is removed."),
        ("What does HTTP 404 mean?", "HTTP 404 means the server was reached but could not find a resource matching the requested path."),
        ("What does HTTP 503 mean?", "HTTP 503 means the service is temporarily unable to handle the request, often because it is overloaded, down, or restarting."),
        ("What is recursion?", "Recursion is a technique where a function solves a problem by calling itself on a smaller case until it reaches a base case."),
        ("What is time complexity?", "Time complexity describes how an algorithm's work grows as its input size grows, often using Big O notation."),
        ("Why use version control?", "Version control records changes, supports collaboration, and makes it possible to compare or restore earlier states."),
        ("What is machine learning?", "Machine learning trains a model from examples so it can detect patterns and make predictions or generate outputs for new inputs."),
        ("What is a language model?", "A language model estimates likely token sequences from context. It can generate fluent text without being conscious or automatically knowing whether every claim is true."),
        ("What is artificial intelligence?", "Artificial intelligence is the field of building computer systems that perform tasks associated with perception, learning, reasoning, language, or decision-making."),
        ("What is overfitting?", "Overfitting happens when a model learns training examples too specifically and performs poorly on genuinely new examples."),
        ("What is the capital of Norway?", "Oslo is the capital of Norway."),
        ("What is the capital of Argentina?", "Buenos Aires is the capital of Argentina."),
        ("What is the capital of South Korea?", "Seoul is the capital of South Korea."),
        ("What is the capital of Kenya?", "Nairobi is the capital of Kenya."),
        ("What is the capital of New Zealand?", "Wellington is the capital of New Zealand."),
        ("Which ocean is largest?", "The Pacific Ocean is the largest ocean on Earth."),
        ("Which planet is closest to the Sun?", "Mercury is the planet closest to the Sun."),
        ("What is the largest mammal?", "The blue whale is the largest known mammal."),
        ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
        ("What language is mainly spoken in Brazil?", "Portuguese is the main language spoken in Brazil."),
        ("How many sides does a hexagon have?", "A hexagon has six sides."),
    ]
    for question, answer in facts:
        rows.append(row(question, answer, "public-v20-general-facts"))
        rows.append(row("Explain simply: " + question.removesuffix("?"), answer, "public-v20-general-facts"))

    # Transferable quantitative and verbal reasoning.  The values and stories
    # deliberately differ from every fixed acceptance question.
    for total in range(12, 53, 4):
        for remain in range(3, min(total, 12), 3):
            rows.append(row(
                f"A shelf held {total} books. All except {remain} were borrowed. How many stayed?",
                f"{remain} books stayed. 'All except {remain}' means those {remain} were not borrowed.",
                "public-v20-general-reasoning",
            ))
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for index, day in enumerate(weekdays):
        prior = weekdays[(index - 2) % 7]
        after = weekdays[(index + 1) % 7]
        rows.append(row(
            f"Two days after {prior} is today. What day comes next?",
            f"Two days after {prior} is {day}, so the next day is {after}.",
            "public-v20-general-reasoning",
        ))
    for start in range(5, 30, 3):
        used = start // 3
        received = (start % 5) + 2
        final = start - used + received
        rows.append(row(
            f"Mina had {start} tokens, spent {used}, then received {received}. How many does she have? Show the steps.",
            f"Start with {start}. Subtract {used} to get {start-used}, then add {received}: {start-used} + {received} = {final}. Mina has {final} tokens.",
            "public-v20-general-reasoning",
        ))
    logic = [
        ("Every tulip is a flower. This plant is a tulip. What follows?", "The plant is a flower, because all tulips belong to the group of flowers."),
        ("Some musicians are teachers. Can we conclude every teacher is a musician?", "No. 'Some musicians are teachers' establishes an overlap, not that every teacher is a musician."),
        ("A switch is either on or off. It is not on. What state is it in?", "It is off, assuming those are the only two possible states."),
        ("Three runners finish in order: Lee is before Sam, and Sam is before Noor. Who is first?", "Lee is first because Lee is before Sam, who is before Noor."),
        ("A bag marked RED contains only blue marbles, and a bag marked BLUE contains only red marbles. Both labels are wrong. How should they be labeled?", "Rename the bag currently marked RED to BLUE, and rename the bag currently marked BLUE to RED."),
        ("Four cards show A, D, 4, and 7. To test 'if a card has a vowel, the other side is even,' which must be turned?", "Turn A to check for an even number and 7 to check that its other side is not a vowel. D and 4 cannot violate the stated direction by themselves."),
    ]
    rows.extend(row(q, a, "public-v20-general-logic") for q, a in logic)

    # Constraint-following is taught as a general skill with unrelated subjects.
    constrained = [
        ("Explain evaporation in exactly two sentences without using the word hot.", "Evaporation happens when molecules at a liquid's surface gain enough energy to enter the gas phase. Wind and low humidity can make the process happen faster."),
        ("Describe a bicycle in one sentence and do not use the words wheel or pedal.", "A bicycle is a human-powered vehicle with two circular supports arranged in line and steered by handlebars."),
        ("Give exactly three reasons to back up files.", "1. Hardware can fail.\n2. Malware or mistakes can erase data.\n3. Versioned copies let you recover earlier work."),
        ("Answer with only yes or no: Is eight an even number?", "Yes"),
        ("Return only the corrected spelling: enviroment", "environment"),
        ("Explain sound in exactly three short sentences without using the words air or ear.", "Sound begins as vibration. The vibration travels through a material as a wave. A listener detects that wave and interprets it."),
        ("Describe a thunderstorm in two sentences without saying rain, cloud, or lightning.", "Dark skies gather while charged regions release brilliant flashes and booming pressure waves. Water then falls heavily as wind moves across the landscape."),
        ("List four primary colors of light or printing, using only a comma-separated line.", "red, green, blue, black"),
    ]
    rows.extend(row(q, a, "public-v20-general-instructions") for q, a in constrained)

    coding = [
        ("This Python fails. Return only fixed code: value = int('7')\nprint(value + '2')", "value = int('7')\nprint(value + 2)"),
        ("Fix this C# assignment and return only code: double price = \"4.5\"; Console.WriteLine(price);", "double price = 4.5;\nConsole.WriteLine(price);"),
        ("Why is this loop infinite? while (i < 10) { Console.WriteLine(i); }", "The loop never changes `i`, so if `i` starts below 10 the condition remains true. Increment it inside the loop, for example with `i++;`."),
        ("Write a short JavaScript function that checks whether a number is even.", "```javascript\nfunction isEven(number) {\n  return number % 2 === 0;\n}\n```"),
        ("Write a short Python function that returns the largest value in a nonempty list.", "```python\ndef largest(values):\n    return max(values)\n```"),
        ("In Unity, why multiply movement by Time.deltaTime?", "It converts a per-second speed into movement for the current frame, making motion much less dependent on frame rate."),
        ("Write a compact Unity component that shows elapsed seconds in a UI text label.", "```csharp\nusing TMPro;\nusing UnityEngine;\n\npublic class TimerLabel : MonoBehaviour\n{\n    [SerializeField] TMP_Text label;\n    void Update() => label.text = Time.time.ToString(\"F1\");\n}\n```\nUpdating complex strings every frame can allocate memory; reduce the update rate when high precision is unnecessary."),
        ("What is the difference between == and = in C#?", "`=` assigns a value, while `==` compares two values for equality."),
        ("What does a null reference error mean?", "Code tried to access a member through a reference that currently points to no object. Find which reference is null and initialize it or guard the access."),
        ("How do I avoid duplicate Discord bot replies?", "Run one bot instance, ignore the bot's own messages, route each incoming message through one handler, and use the message ID as an idempotency key while it is being processed."),
    ]
    rows.extend(row(q, a, "public-v20-general-code") for q, a in coding)

    creativity = [
        ("Invent a silly ability that is useless most of the time but saves one picnic.", "You can make exactly one napkin hover three centimeters above a table. It becomes heroic when a sudden puddle races toward the only napkin holding the handwritten directions home."),
        ("Describe a stalled printer like a sports commentator.", "The page enters the arena—no, it stops at the gate! The printer flashes PAPER JAM while the crowd reaches for the access panel in stunned silence."),
        ("Describe a phone at one percent battery like a medieval prophecy.", "When the final sliver of the glowing sigil fades, the kingdom shall lose its maps, its messages, and—most terrible of all—the unfinished grocery list."),
        ("Invent a harmless superpower useful only in a library emergency.", "You can silence one squeaky chair for twelve seconds, which is perfect when a student must move it during the quietest moment of a final exam."),
        ("Write a two-line joke about debugging.", "I removed the last bug from my code.\nNow the program cannot find the feature it was living in."),
    ]
    rows.extend(row(q, a, "public-v20-general-creativity") for q, a in creativity)

    # Short multi-turn examples teach reference tracking rather than a canned
    # response to one wording.
    rows.extend([
        {"messages": [
            {"role": "user", "content": "Call the first planet Ember and the second planet Tide."},
            {"role": "assistant", "content": "Got it: the first planet is Ember and the second is Tide."},
            {"role": "user", "content": "Which one was second?"},
            {"role": "assistant", "content": "Tide was the second planet."},
        ], "source": "public-v20-general-context"},
        {"messages": [
            {"role": "user", "content": "Explain what RAM is."},
            {"role": "assistant", "content": "RAM is fast temporary storage used by programs that are currently running."},
            {"role": "user", "content": "Explain that to a ten-year-old."},
            {"role": "assistant", "content": "RAM is like a desk where your computer keeps the things it is using right now. A bigger desk lets it keep more things ready without putting them away first."},
        ], "source": "public-v20-general-context"},
    ])

    # Deterministic order, with exact duplicate conversations removed.
    unique: dict[str, dict[str, object]] = {}
    for item in rows:
        key = json.dumps(item["messages"], ensure_ascii=False, sort_keys=True)
        unique[key] = item
    built = list(unique.values())
    random.Random(20260906).shuffle(built)
    return built


def main() -> None:
    rows = build_rows()
    OUTPUT.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows):,} independent general-knowledge examples to {OUTPUT}")


if __name__ == "__main__":
    main()
