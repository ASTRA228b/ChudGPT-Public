"""Build broad Python and C# supervised curricula for Public V20."""

from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_OUTPUT = ROOT / "data/public_python_csharp_curriculum.jsonl"
MIXED_OUTPUT = ROOT / "data/public_k8_code_curriculum.jsonl"
K8_INPUT = ROOT / "data/public_k8_curriculum.jsonl"


def main() -> None:
    rng = random.Random(20260909)
    rows: list[dict[str, object]] = []

    def add(language: str, topic: str, prompt: str, answer: str) -> None:
        rows.append({
            "source": "chudgpt-public-code-v1",
            "category": f"{language}_{topic}",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
        })

    variables = ["value", "score", "count", "total", "speed", "health", "level", "amount"]
    functions = ["adjust", "update", "calculate", "transform", "increase", "scale", "offset", "modify"]
    for index in range(320):
        variable = variables[index % len(variables)]
        function = f"{functions[(index * 3) % len(functions)]}_{index}"
        number = 2 + index % 29
        operation, symbol = [("adds", "+"), ("subtracts", "-"), ("multiplies by", "*")][index % 3]
        wording = f"subtracts {number} from" if symbol == "-" else f"{operation} {number} to" if symbol == "+" else f"multiplies"
        if symbol == "*":
            description = f"multiplies its {variable} argument by {number}"
        elif symbol == "+":
            description = f"adds {number} to its {variable} argument"
        else:
            description = f"subtracts {number} from its {variable} argument"
        add("python", "functions", f"Write a typed Python function named {function} that {description}. Code only.",
            f"def {function}({variable}: float) -> float:\n    return {variable} {symbol} {number}")
        add("csharp", "methods", f"Write a C# method named {function.title().replace('_', '')} that {description}. Code only.",
            f"static float {function.title().replace('_', '')}(float {variable})\n{{\n    return {variable} {symbol} {number};\n}}")

    collections = ["scores", "values", "prices", "temperatures", "distances", "ages", "levels", "times"]
    for index in range(280):
        name = collections[index % len(collections)]
        threshold = 1 + (index * 7) % 50
        add("python", "collections", f"Python: return every item in {name} greater than {threshold}. Use a list comprehension. Code only.",
            f"result = [item for item in {name} if item > {threshold}]")
        add("csharp", "linq", f"C#: create a list containing every value in {name} greater than {threshold}. Use LINQ. Code only.",
            f"List<int> result = {name}.Where(value => value > {threshold}).ToList();")
    for index in range(180):
        name = collections[index % len(collections)]
        add("python", "collections", f"Write a Python function that safely returns the first item of {name}, or None when it is empty.",
            f"def first_{name}({name}):\n    return {name}[0] if {name} else None")
        add("csharp", "collections", f"Write a C# method that returns the first integer in {name}, or null when it is empty.",
            f"static int? First{ name.title() }(IReadOnlyList<int> {name})\n{{\n    return {name}.Count > 0 ? {name}[0] : null;\n}}")

    words = ["message", "username", "filename", "title", "sentence", "label", "command", "description"]
    for index in range(200):
        name = words[index % len(words)]
        add("python", "strings", f"Python function that trims whitespace from {name} and converts it to lowercase. Code only.",
            f"def normalize_{name}({name}: str) -> str:\n    return {name}.strip().lower()")
        add("csharp", "strings", f"C# method that trims whitespace from {name} and converts it to lowercase. Code only.",
            f"static string Normalize{name.title()}(string {name})\n{{\n    return {name}.Trim().ToLowerInvariant();\n}}")

    for index in range(140):
        class_name = ["Player", "Enemy", "Book", "Song", "Vehicle", "Account", "Quest"][index % 7] + str(index)
        field = variables[(index * 5) % len(variables)]
        add("python", "classes", f"Create a small Python dataclass named {class_name} with a string name and integer {field}.",
            f"from dataclasses import dataclass\n\n@dataclass\nclass {class_name}:\n    name: str\n    {field}: int")
        add("csharp", "classes", f"Create a C# class named {class_name} with public read-only Name and {field.title()} properties set by its constructor.",
            f"public class {class_name}\n{{\n    public string Name {{ get; }}\n    public int {field.title()} {{ get; }}\n\n    public {class_name}(string name, int {field})\n    {{\n        Name = name;\n        {field.title()} = {field};\n    }}\n}}")

    for index in range(160):
        path = ["scores.txt", "config.txt", "notes.txt", "levels.txt"][index % 4]
        add("python", "files", f"Python: read all text from '{path}' using UTF-8 and a context manager.",
            f"with open(\"{path}\", \"r\", encoding=\"utf-8\") as file:\n    text = file.read()")
        add("csharp", "files", f"C#: asynchronously read all text from '{path}'. Code only.",
            f"string text = await File.ReadAllTextAsync(\"{path}\");")

    for index in range(180):
        divisor = 1 + index % 20
        add("python", "exceptions", f"Write a Python function safe_divide_{index}(a) that divides a by {divisor} and raises TypeError unless a is int or float.",
            f"def safe_divide_{index}(a):\n    if not isinstance(a, (int, float)):\n        raise TypeError(\"a must be numeric\")\n    return a / {divisor}")
        add("csharp", "exceptions", f"Write a C# method SafeDivide{index}(double value, double divisor) that throws when divisor is zero.",
            f"static double SafeDivide{index}(double value, double divisor)\n{{\n    if (divisor == 0)\n        throw new DivideByZeroException();\n    return value / divisor;\n}}")

    for index in range(260):
        value = 2 + index % 90
        broken_python = rng.choice([
            f"number = '{value}'\nprint(number + 1)",
            f"values = [{value}, {value + 1}]\nprint(values[2])",
            f"def double(number)\n    return number * 2",
        ])
        if "number = '" in broken_python:
            fixed = f"number = {value}\nprint(number + 1)"
            reason = "The quotes made number a string; use an integer before adding."
        elif "values" in broken_python:
            fixed = f"values = [{value}, {value + 1}]\nprint(values[1])"
            reason = "A two-item list has indexes 0 and 1, so index 2 is out of range."
        else:
            fixed = "def double(number):\n    return number * 2"
            reason = "A Python function definition needs a colon after its signature."
        add("python", "debugging", f"Fix this Python and explain the bug:\n```python\n{broken_python}\n```",
            f"```python\n{fixed}\n```\n{reason}")

        broken_csharp = rng.choice([
            f'int number = "{value}";\nConsole.WriteLine(number + 1);',
            f"int[] values = {{ {value}, {value + 1} }};\nConsole.WriteLine(values[2]);",
            "if (ready)\n    Console.WriteLine(\"Ready\")",
        ])
        if 'int number = "' in broken_csharp:
            fixed_cs = f"int number = {value};\nConsole.WriteLine(number + 1);"
            reason_cs = "An int cannot be initialized with a string literal; remove the quotes."
        elif "values[2]" in broken_csharp:
            fixed_cs = f"int[] values = {{ {value}, {value + 1} }};\nConsole.WriteLine(values[1]);"
            reason_cs = "A two-element array has indexes 0 and 1, so index 2 is out of range."
        else:
            fixed_cs = "if (ready)\n    Console.WriteLine(\"Ready\");"
            reason_cs = "C# statements need a terminating semicolon."
        add("csharp", "debugging", f"Fix this C# and explain the bug:\n```csharp\n{broken_csharp}\n```",
            f"```csharp\n{fixed_cs}\n```\n{reason_cs}")

    for index in range(160):
        function = f"is_even_{index}"
        add("python", "testing", f"Write a Python function {function}(number) and one pytest test showing that {2 * (index % 20)} is even.",
            f"def {function}(number: int) -> bool:\n    return number % 2 == 0\n\ndef test_{function}():\n    assert {function}({2 * (index % 20)}) is True")
        add("csharp", "testing", f"Write a C# method IsEven{index}(int number) and one NUnit assertion showing that {2 * (index % 20)} is even.",
            f"static bool IsEven{index}(int number) => number % 2 == 0;\n\nAssert.That(IsEven{index}({2 * (index % 20)}), Is.True);")

    for index in range(180):
        speed = 2 + index % 18
        add("csharp", "unity", f"Unity C#: move a Rigidbody named body forward at {speed} units per second in FixedUpdate. Use MovePosition and fixedDeltaTime.",
            f"[SerializeField] private Rigidbody body;\n\nprivate void FixedUpdate()\n{{\n    Vector3 offset = transform.forward * {speed}f * Time.fixedDeltaTime;\n    body.MovePosition(body.position + offset);\n}}")
        add("csharp", "unity", f"Unity C#: expose an integer score in the Inspector and add {index % 10 + 1} to it in a public method.",
            f"[SerializeField] private int score;\n\npublic void AddScore()\n{{\n    score += {index % 10 + 1};\n}}")
    for index in range(120):
        seconds = 1 + index % 15
        add("csharp", "async", f"C#: write an async method Wait{index}Async that waits {seconds} seconds without blocking the thread.",
            f"static async Task Wait{index}Async()\n{{\n    await Task.Delay(TimeSpan.FromSeconds({seconds}));\n}}")
        add("python", "async", f"Python: write async function wait_{index} that waits {seconds} seconds without blocking the event loop.",
            f"import asyncio\n\nasync def wait_{index}():\n    await asyncio.sleep({seconds})")

    explanations = [
        ("python", "Why use a context manager when opening a file?", "A context manager closes the file reliably even if an exception occurs."),
        ("python", "When should you use a dictionary instead of a list?", "Use a dictionary when values should be retrieved by meaningful unique keys rather than numeric positions."),
        ("python", "What does a Python generator do?", "A generator yields values one at a time, which can avoid storing the entire sequence in memory."),
        ("python", "Why avoid a mutable list as a default argument?", "The same list object is reused across calls, so changes can leak between calls. Use None and create a list inside."),
        ("csharp", "What is the difference between a class and a struct in C#?", "A class is a reference type; a struct is a value type. Structs work best for small value-like data that does not need inheritance."),
        ("csharp", "Why use an interface in C#?", "An interface defines a contract that different types can implement, reducing coupling between callers and concrete classes."),
        ("csharp", "Why use FixedUpdate for Rigidbody movement in Unity?", "FixedUpdate runs with the physics timestep, which keeps physics changes synchronized and more consistent."),
        ("csharp", "What does async and await do in C#?", "They let a method pause until asynchronous work finishes without blocking the calling thread, then continue when the result is ready."),
    ]
    for repeat in range(35):
        for language, prompt, answer in explanations:
            add(language, "explanation", prompt, answer)

    rng.shuffle(rows)
    CODE_OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    if not K8_INPUT.is_file():
        raise FileNotFoundError(f"Build the K-8 curriculum first: {K8_INPUT}")
    combined = K8_INPUT.read_text(encoding="utf-8") + CODE_OUTPUT.read_text(encoding="utf-8")
    MIXED_OUTPUT.write_text(combined, encoding="utf-8")
    print(f"Wrote {len(rows):,} Python/C# examples to {CODE_OUTPUT}")
    print(f"Combined K-8/code curriculum: {sum(1 for line in combined.splitlines() if line.strip()):,} examples")


if __name__ == "__main__":
    main()
