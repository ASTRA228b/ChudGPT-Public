"""Reviewed conversation/development curriculum; never reads benchmark answers."""
import json
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build(binding=False):
    rows = []
    def add(category, *turns):
        rows.append({"source": "reviewed_balanced_v1", "category": category,
                     "messages": [{"role": "user" if i % 2 == 0 else "assistant", "content": value}
                                  for i, value in enumerate(turns)]})
    pairs = [
        ("I'm bored.", "Want to make something ridiculous, talk about games, or try a small coding challenge?"),
        ("I finally finished my project!", "Nice! What did you build?"),
        ("My game keeps crashing.", "What error appears, and what were you doing just before it crashed? Send the relevant code and we can trace it."),
        ("I don't feel like doing anything today.", "Sounds like a low-energy day. Want to talk, or pick one tiny thing to get done?"),
        ("Tell me something interesting.", "An octopus has three hearts. Apparently one heart wasn't enough paperwork."),
        ("Can we just talk?", "Of course. What's been on your mind?"),
        ("I'm learning to code.", "What language are you learning, and what would you like to build with it?"),
        ("That didn't help.", "Which part missed what you needed? Show me the result you expected and I'll try a more direct approach."),
        ("Make that shorter.", "Sure. Which text should I shorten?"),
        ("What should I build this weekend?", "Try a tiny game with one mechanic: dodge falling objects, collect coins, or solve a short puzzle."),
        ("My code finally works.", "The bugs have temporarily agreed to a ceasefire. What was the fix?"),
        ("I like making weird games.", "Weird games are a great excuse to experiment. What's your strangest mechanic so far?"),
        ("I am tired after school.", "That sounds exhausting. Are you taking a break or trying to get something finished?"),
        ("My project is too big.", "Pick one small playable feature first. Finish that, then add the next feature instead of building everything at once."),
        ("Should I learn everything at once?", "Choose one small goal and learn what it needs. You can expand as your projects grow."),
        ("I made a spelling mistake.", "Happens. Send the sentence if you'd like help fixing it."),
    ]
    for q, a in pairs: add("conversation", q, a)
    for name, color, item in [("Luma", "violet", "drone"), ("Tango", "orange", "cat"),
                              ("Nimbus", "silver", "robot"), ("Pip", "teal", "spaceship"),
                              ("Moss", "green", "character"), ("Ember", "red", "dragon")]:
        add("memory", f"My {item} is named {name}.", f"Got it: your {item} is {name}.",
            "What did I name it?", f"You named it {name}.")
        add("memory", f"My favorite color is {color}.", f"Got it, {color}.",
            "What color did I say?", f"You said {color}.")
        add("correction", f"Call my project {name}.", f"Your project is {name}.",
            f"Actually, rename it {name} Two.", f"Okay, the project is now {name} Two.",
            "What's the current name?", f"The current name is {name} Two.")
    add("context", "I'm making a racing game.", "What part are you working on?",
        "The controls.", "For racing controls, start with steering, acceleration, and braking. Tune those before adding extra mechanics.")
    add("context", "I'm learning Python.", "What would you like to try?", "Loops.",
        "A for loop repeats work for each item. For example:\n```python\nfor animal in ['cat', 'dog']:\n    print(animal)\n```")
    add("context", "Tell me about trees.", "Trees use sunlight to make sugars and take up water through their roots.",
        "Now let's talk about databases.", "Sure. A database stores information so programs can query and update it.",
        "Which topic are we on?", "We're talking about databases now.")
    add("context", "Pick a name for a puzzle game.", "How about Hidden Lantern?", "Why that name?",
        "Hidden suggests discovery, and Lantern suggests finding your way. Both fit a puzzle game.")
    add("context", "A stack removes the most recently added item first.", "Right, that is last in, first out.",
        "Explain it more simply.", "Think of a pile of plates: you take the top plate off first.")
    code = [
        ("Write Python that returns the length of a list.", "def list_length(items):\n    return len(items)"),
        ("Make a Python function that adds two numbers.", "def add(a, b):\n    return a + b"),
        ("Python function to check whether a number is even.", "def is_even(number):\n    return number % 2 == 0"),
        ("Return the first item of a Python list, or None if empty.", "def first(items):\n    return items[0] if items else None"),
        ("Write a function to uppercase a string in Python.", "def uppercase(text):\n    return text.upper()"),
        ("Count positive numbers in Python.", "def count_positive(numbers):\n    return sum(1 for number in numbers if number > 0)"),
        ("Remove duplicates from a Python list while keeping order.", "def unique(items):\n    return list(dict.fromkeys(items))"),
        ("Return the last three characters of a string in Python.", "def ending(text):\n    return text[-3:]"),
        ("Join a list of words using spaces in Python.", "def sentence(words):\n    return ' '.join(words)"),
        ("Write a Python function to double each list item.", "def double_all(numbers):\n    return [number * 2 for number in numbers]"),
    ]
    for q, code_text in code:
        add("python", q, "```python\n" + code_text + "\n```")
        add("python_code_only", q + " Return code only, without Markdown.", code_text)
    for lang, q, a in [
        ("javascript", "JavaScript function to multiply two numbers.", "function multiply(a, b) {\n  return a * b;\n}"),
        ("javascript", "Filter negative values from a JavaScript array.", "const nonnegative = values.filter(value => value >= 0);"),
        ("javascript", "Use JavaScript to read text from an element with id score.", "const score = document.getElementById('score').textContent;"),
        ("java", "In Java, print each element of an integer array.", "for (int value : values) {\n    System.out.println(value);\n}"),
        ("java", "Java method that adds two integers.", "static int add(int a, int b) {\n    return a + b;\n}"),
        ("java", "Java while loop counting from 0 through 2.", "int i = 0;\nwhile (i < 3) {\n    System.out.println(i);\n    i++;\n}"),
        ("csharp", "C# method to square a float.", "static float Square(float value)\n{\n    return value * value;\n}"),
        ("csharp", "C# foreach loop printing each name.", "foreach (string name in names)\n{\n    Console.WriteLine(name);\n}"),
        ("sql", "SQL: select names from users older than 18.", "SELECT name FROM users WHERE age > 18;"),
        ("sql", "SQL: count rows in orders.", "SELECT COUNT(*) FROM orders;"),
        ("css", "CSS to make a class called warning red.", ".warning {\n  color: red;\n}"),
        ("html", "HTML link to /about labeled About.", '<a href="/about">About</a>'),
    ]: add("development", q, f"```{lang}\n{a}\n```")
    for q, a in [
        ("Why does Python range(4) stop at 3?", "The stop value is exclusive. range(4) produces 0, 1, 2, and 3."),
        ("What is a null reference error?", "The program tried to use an object reference that has no object. Check where that reference should be assigned before accessing it."),
        ("What is an off-by-one error?", "A loop or index goes one position too far or stops one position too soon. Check its starting value and whether the end comparison is inclusive."),
        ("How should I debug a crash?", "Read the error and stack trace, reproduce the crash, then inspect the first relevant line in your code. Change one cause at a time and test again."),
        ("What does a unit test do?", "A unit test checks a small part of a program against an expected result. It helps catch mistakes when code changes."),
        ("Why avoid creating objects every frame?", "Frequent allocation can increase garbage collection and cause frame-time spikes. Reuse objects when appropriate and measure with a profiler."),
        ("What is a race condition?", "The result depends on the timing of concurrent operations accessing shared state. Synchronization or avoiding shared mutable state can prevent it."),
        ("What is the difference between a list and a dictionary?", "A list stores items in sequence and uses positions to access them. A dictionary maps keys to values."),
        ("What is the difference between a parameter and an argument?", "A parameter is a variable declared by a function. An argument is the value supplied when calling it."),
        ("What is a compiler?", "A compiler translates source code into another form, often machine code or intermediate code, and reports errors it finds during compilation."),
        ("Why do leaves look green?", "Chlorophyll absorbs red and blue light more strongly and reflects or transmits more green light."),
        ("Why does metal feel colder than wood?", "Metal carries heat away from your hand faster than wood, even when both are at the same room temperature."),
        ("What is condensation?", "Condensation is water vapor turning into liquid droplets, often when moist air cools."),
        ("What does a battery store?", "A battery stores chemical energy and converts it into electrical energy through chemical reactions."),
        ("What is a food chain?", "A food chain shows how energy passes between organisms when one eats another."),
        ("What is the difference between weather and climate?", "Weather describes current or short-term atmospheric conditions. Climate describes patterns over many years."),
        ("Why do we use paragraphs?", "Paragraphs group related ideas so readers can follow the structure of a text."),
        ("How do I make a plan manageable?", "Define a small result, split it into clear steps, and finish the most important step first."),
    ]: add("explanation", q, a)
    for q, a in [
        ("Summarize: The shop opens at nine. It closes at five. It sells books.", "The bookshop is open from nine to five."),
        ("Summarize: Nina missed the bus. She walked instead. She arrived on time.", "Nina walked after missing the bus and still arrived on time."),
        ("Summarize: Rain filled the tank. The stored water was used on the garden.", "Collected rainwater was used to water the garden."),
        ("Return only the city from this text: Mara lives in Lisbon.", "Lisbon"),
        ("Return only the animal: A brown fox crossed the road.", "fox"),
        ("Name two colors, separated by a comma.", "blue, yellow"),
        ("Explain a loop in one sentence.", "A loop repeats a set of instructions."),
        ("Give exactly three words describing snow.", "Cold white flakes"),
    ]: add("instructions", q, a)
    # Teach copying and variable binding across many independent values. These
    # are supervised examples only; none is a runtime answer lookup.
    rng = random.Random(901)
    starts = ["Al", "Ben", "Cor", "Del", "Ev", "Fen", "Gal", "Har", "Iv", "Jen", "Kel", "Lor"]
    ends = ["ba", "do", "fi", "go", "la", "mi", "no", "pa", "ra", "si", "ta", "vo"]
    names = [a + b for a in starts for b in ends]
    rng.shuffle(names)
    if not binding:
        names = []
    for index, name in enumerate(names):
        item = ["robot", "pet", "game", "project", "character", "boat"][index % 6]
        prompts = [f"My {item} is named {name}.", f"I named my {item} {name}.",
                   f"The name of my {item} is {name}."]
        followups = [f"What is my {item} called?", "What name did I tell you?", "Can you repeat its name?"]
        add("binding", prompts[index % 3], f"Your {item} is called {name}.",
            followups[index % 3], f"Its name is {name}.")
        add("extraction", f"Return only the name from this sentence: The pilot is {name}.", name)
    for index in range(120 if binding else 0):
        identifier = f"compute_{index}"
        variable = ["value", "amount", "n", "x", "input_value"][index % 5]
        number = 2 + index % 11
        operation, symbol = [("adds", "+"), ("multiplies by", "*"), ("subtracts", "-")][index % 3]
        if symbol == "-":
            instruction = f"subtracts {number} from {variable}"
        elif symbol == "*":
            instruction = f"multiplies {variable} by {number}"
        else:
            instruction = f"adds {number} to {variable}"
        add("code_binding", f"Define a Python function {identifier}({variable}) that {instruction}. Code only.",
            f"def {identifier}({variable}):\n    return {variable} {symbol} {number}")
    for index in range(60 if binding else 0):
        array = ["names", "items", "words", "entries", "titles", "messages"][index % 6]
        variable = ["item", "entry", "text", "name", "word"][index % 5]
        add("java_binding", f"Java: print each String in the array {array}, using a variable called {variable}.",
            f"```java\nfor (String {variable} : {array}) {{\n    System.out.println({variable});\n}}\n```")
    # Expand multi-turn records into supervised prefixes so every assistant
    # turn, not just the final answer, is trained exactly once.
    expanded = []
    for record in rows:
        for end in range(2, len(record["messages"]) + 1, 2):
            expanded.append({**record, "messages": record["messages"][:end]})
    reviewed = [json.loads(line) for line in (ROOT / "data/public_v20_general_knowledge.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expanded.extend(reviewed)
    unique = {json.dumps(r["messages"], sort_keys=True): r for r in expanded}
    destination = ROOT / ("data/public_balanced_binding.jsonl" if binding else "data/public_balanced_reviewed.jsonl")
    destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in unique.values()), encoding="utf-8")
    print(f"Wrote {len(unique)} reviewed examples to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", action="store_true")
    build(parser.parse_args().binding)
