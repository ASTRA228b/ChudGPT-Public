"""Build the post-foundation SFT mix without legacy response-loop corpora."""

from __future__ import annotations

import json
import random
from pathlib import Path

from build_public_extended_corpus import load_rows


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/public_quality_sft.jsonl"
CODE_OUTPUT = ROOT / "data/public_reviewed_python_csharp.jsonl"


PYTHON_TASKS = [
    ("Write Python code that reverses a string named text.", "reversed_text = text[::-1]"),
    ("Define reverse_words(text) in Python to reverse space-separated word order.", "def reverse_words(text: str) -> str:\n    return \" \".join(reversed(text.split()))"),
    ("Write Python that counts negative values in numbers.", "negative_count = sum(1 for number in numbers if number < 0)"),
    ("Write a Python function that returns only even numbers from values.", "def even_numbers(values: list[int]) -> list[int]:\n    return [value for value in values if value % 2 == 0]"),
    ("Write a Python palindrome checker.", "def is_palindrome(text: str) -> bool:\n    cleaned = \"\".join(char.lower() for char in text if char.isalnum())\n    return cleaned == cleaned[::-1]"),
    ("Write a Python function that counts word frequencies.", "def word_counts(words: list[str]) -> dict[str, int]:\n    counts: dict[str, int] = {}\n    for word in words:\n        counts[word] = counts.get(word, 0) + 1\n    return counts"),
    ("Read a UTF-8 text file named notes.txt in Python.", "with open(\"notes.txt\", \"r\", encoding=\"utf-8\") as file:\n    text = file.read()"),
    ("Safely parse a Python integer and return None if parsing fails.", "def parse_integer(text: str) -> int | None:\n    try:\n        return int(text)\n    except ValueError:\n        return None"),
    ("Write Python to sort players by their score key descending.", "players.sort(key=lambda player: player[\"score\"], reverse=True)"),
    ("Write a typed Python clamp(value, minimum, maximum) function.", "def clamp(value: float, minimum: float, maximum: float) -> float:\n    return max(minimum, min(value, maximum))"),
    ("Write a Python generator yielding squares from 0 through limit - 1.", "def squares(limit: int):\n    for number in range(limit):\n        yield number * number"),
    ("Write an async Python function that waits one second without blocking.", "import asyncio\n\nasync def wait_one_second() -> None:\n    await asyncio.sleep(1)"),
    ("Fix this Python: def double(number) return number * 2", "def double(number):\n    return number * 2"),
    ("Fix this Python IndexError: items = ['a', 'b']; print(items[2])", "items = [\"a\", \"b\"]\nprint(items[1])"),
    ("Why is [] risky as a Python default argument?", "The same mutable list is reused across calls, so changes persist. Use None and create a new list inside the function."),
    ("Explain a Python dictionary to a beginner.", "A dictionary stores values under unique keys. You use a key, such as a username, to quickly retrieve its associated value."),
]


CSHARP_TASKS = [
    ("Write C# ClampScore(int score) that keeps a score from 0 to 100.", "static int ClampScore(int score)\n{\n    return Math.Clamp(score, 0, 100);\n}"),
    ("Use C# LINQ to make a list of even values from numbers.", "List<int> evens = numbers.Where(number => number % 2 == 0).ToList();"),
    ("Write a C# method that reverses a string.", "static string ReverseText(string text)\n{\n    return new string(text.Reverse().ToArray());\n}"),
    ("Write a C# method that safely parses an integer.", "static int? ParseInteger(string text)\n{\n    return int.TryParse(text, out int value) ? value : null;\n}"),
    ("Write C# that asynchronously reads notes.txt.", "string text = await File.ReadAllTextAsync(\"notes.txt\");"),
    ("Write a C# interface named IDamageable with a TakeDamage method.", "public interface IDamageable\n{\n    void TakeDamage(int amount);\n}"),
    ("Write a small C# record for a player name and score.", "public record Player(string Name, int Score);"),
    ("Write a C# method returning the largest integer in values.", "static int Largest(IEnumerable<int> values)\n{\n    return values.Max();\n}"),
    ("Write C# to count negative values in numbers.", "int negativeCount = numbers.Count(number => number < 0);"),
    ("Write an async C# method that waits one second without blocking.", "static async Task WaitOneSecondAsync()\n{\n    await Task.Delay(TimeSpan.FromSeconds(1));\n}"),
    ("Fix this C# code: int x = \"10\"; Console.WriteLine(x + 5);", "int x = 10;\nConsole.WriteLine(x + 5);"),
    ("Fix the missing C# semicolon: Console.WriteLine(\"Ready\")", "Console.WriteLine(\"Ready\");"),
    ("Why use an interface in C#?", "An interface defines a contract that different types can implement. It lets callers depend on shared behavior instead of one concrete class."),
    ("Explain class versus struct in C#.", "A class is a reference type, while a struct is a value type. Structs are best for small value-like data; classes suit objects with identity or inheritance."),
    ("In Unity, why move a Rigidbody in FixedUpdate?", "FixedUpdate follows Unity's physics timestep, keeping Rigidbody movement synchronized with the physics simulation."),
    ("Unity C#: expose score in the Inspector and add 10 in AddScore.", "[SerializeField] private int score;\n\npublic void AddScore()\n{\n    score += 10;\n}"),
    ("Unity C#: move Rigidbody body forward at 5 units per second.", "[SerializeField] private Rigidbody body;\n\nprivate void FixedUpdate()\n{\n    Vector3 offset = transform.forward * 5f * Time.fixedDeltaTime;\n    body.MovePosition(body.position + offset);\n}"),
    ("Write a Unity C# FPS display component under 25 lines.", "using UnityEngine;\nusing UnityEngine.UI;\n\npublic class FpsDisplay : MonoBehaviour\n{\n    [SerializeField] private Text label;\n\n    private void Update()\n    {\n        float fps = 1f / Time.unscaledDeltaTime;\n        label.text = $\"FPS: {fps:0}\";\n    }\n}"),
]


def code_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    variants = (
        "{prompt}",
        "Please help with this development task: {prompt}",
        "Return a concise correct solution. {prompt}",
        "I'm learning. {prompt}",
    )
    for language, tasks in (("python", PYTHON_TASKS), ("csharp", CSHARP_TASKS)):
        for prompt, answer in tasks:
            for template in variants:
                rows.append({
                    "source": "chudgpt-public-reviewed-code-v1",
                    "category": language,
                    "messages": [
                        {"role": "user", "content": template.format(prompt=prompt)},
                        {"role": "assistant", "content": answer},
                    ],
                })
    return rows


def main() -> None:
    sources = (
        ROOT / "data/external/databricks_dolly_15k/databricks-dolly-15k.jsonl",
        ROOT / "data/public_balanced_reviewed.jsonl",
        ROOT / "data/public_balanced_binding.jsonl",
        ROOT / "data/public_k8_curriculum.jsonl",
        ROOT / "data/public_dialogue_life_curriculum.jsonl",
    )
    rows: list[dict[str, object]] = []
    for source in sources:
        rows.extend(load_rows(source))
    reviewed_code = code_rows()
    rows.extend(reviewed_code)
    random.Random(20260911).shuffle(rows)
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    CODE_OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed_code),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows):,} quality SFT conversations and {len(reviewed_code):,} reviewed code variants")


if __name__ == "__main__":
    main()
