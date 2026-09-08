"""Build a concentrated, reviewed capability-polish set for Public V20."""

from __future__ import annotations

import json
import random
from pathlib import Path

from build_public_extended_corpus import load_rows


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/public_core_polish.jsonl"

MATH_CATEGORIES = {
    "arithmetic", "integers", "word_problems", "algebra", "place_value",
    "geometry", "decimals", "statistics", "multiplication", "division",
    "fractions", "percent", "ratios", "probability", "order_of_operations",
}

REVIEWED_KNOWLEDGE = (
    ("science", "What causes the seasons?", "Earth's axis is tilted. As Earth orbits the Sun, each hemisphere receives different angles and lengths of sunlight during the year."),
    ("science", "Why is summer warmer than winter?", "In summer, that hemisphere tilts toward the Sun, giving it more direct sunlight and longer days. Winter sunlight is less direct and days are shorter."),
    ("science", "How are weathering and erosion different?", "Weathering breaks rock into smaller pieces. Erosion moves those pieces by water, wind, ice, or gravity."),
    ("science", "What do lungs do with oxygen?", "Air reaches tiny lung sacs called alveoli. Oxygen enters nearby blood, and the blood carries it to cells throughout the body."),
    ("science", "How does blood work with the lungs?", "Blood picks up oxygen in the lungs and releases carbon dioxide there. The heart then pumps oxygen-rich blood around the body."),
    ("science", "How does pollination help flowering plants?", "Pollination moves pollen to the flower parts that can fertilize an ovule, allowing the plant to make seeds and often fruit."),
    ("science", "Why do objects fall toward Earth?", "Earth's gravity accelerates objects toward its center. Air resistance can slow the fall, especially for wide or light objects."),
    ("science", "What is photosynthesis?", "Plants use light energy to turn water and carbon dioxide into sugars. Oxygen is released as part of the process."),
    ("science", "What changes when ice melts?", "The water changes state from solid to liquid. It is still the same substance, so melting is a physical change."),
    ("science", "Why does the Moon appear to change shape?", "The Moon's shape does not change. As it orbits Earth, we see different portions of its sunlit half."),
    ("technology", "Explain the basic job of a refrigerator.", "A refrigerator uses a circulating refrigerant to absorb heat inside the cabinet and release that heat outside. A compressor and expansion device keep the cycle moving."),
    ("technology", "What is the main difference between TCP and UDP?", "TCP establishes a connection and checks delivery and order. UDP sends independent packets with less overhead but does not guarantee delivery or ordering."),
    ("technology", "Why can a GPU perform many similar calculations at once?", "A GPU contains many smaller processing units built for parallel work. A CPU has fewer, more flexible cores optimized for varied tasks and quick decisions."),
    ("technology", "What does computer memory or RAM do?", "RAM holds data and instructions that active programs need quickly. Its contents normally disappear when power is removed."),
    ("technology", "Why is a software backup useful?", "A backup is a separate copy that can restore data after deletion, corruption, device failure, theft, or ransomware."),
    ("digital_literacy", "How should I check a surprising claim before reposting it?", "Find the original source, inspect its evidence and date, and compare it with independent reliable sources before sharing."),
    ("digital_literacy", "A post has thousands of likes. Does that prove it is true?", "No. Popularity is not evidence. Check the original source, supporting evidence, date, and independent reporting."),
    ("digital_literacy", "What should I do with an unexpected password-reset link?", "Do not use the link. Open the official app or type the known website address yourself, then check account activity and security settings."),
    ("reading", "A backup restored Jo's file after the laptop crashed. What saved the work?", "The backup copy saved the work by providing another copy after the laptop's original file was lost."),
    ("reading", "Rain loosened rocks above a path, crews cleared them, and the path reopened. Why was it closed?", "It was closed because heavy rain loosened rocks and made the path unsafe."),
    ("reading", "Lee missed the bus, borrowed a bicycle, and arrived before class. How did Lee reach school?", "Lee rode a borrowed bicycle to school."),
    ("summary", "Summarize this: A storm knocked out power. Crews repaired a damaged line overnight. Electricity returned in the morning.", "Crews restored power in the morning after repairing storm damage overnight."),
    ("summary", "Summarize this: The library added weekend hours after a community survey. Saturday attendance doubled in one month.", "Weekend hours introduced after a survey helped the library double Saturday attendance."),
    ("summary", "Summarize this: Ana tested three designs. The second used the least material while holding the most weight, so she selected it.", "Ana chose the second design because it held the most weight with the least material."),
)


def reviewed_knowledge_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prefixes = ("", "Explain clearly: ", "Give a short accurate answer: ", "Help a middle-school student: ")
    for category, prompt, answer in REVIEWED_KNOWLEDGE:
        for prefix in prefixes:
            rows.append({
                "source": "chudgpt-public-reviewed-knowledge-v1",
                "category": category,
                "messages": [
                    {"role": "user", "content": prefix + prompt},
                    {"role": "assistant", "content": answer},
                ],
            })
    return rows


def main() -> None:
    rows = reviewed_knowledge_rows()
    for path in (
        ROOT / "data/public_balanced_binding.jsonl",
        ROOT / "data/public_dialogue_life_curriculum.jsonl",
        ROOT / "data/public_reviewed_python_csharp.jsonl",
    ):
        rows.extend(load_rows(path))

    # Keep the broad non-math school material intact. Sample math categories so
    # arithmetic does not crowd out language, science, reasoning, or code.
    math_seen: dict[str, int] = {}
    for item in load_rows(ROOT / "data/public_k8_curriculum.jsonl"):
        category = str(item.get("category", ""))
        if category in MATH_CATEGORIES:
            count = math_seen.get(category, 0)
            if count >= 80:
                continue
            math_seen[category] = count + 1
        rows.append(item)

    unique: list[dict[str, object]] = []
    fingerprints: set[str] = set()
    for item in rows:
        fingerprint = json.dumps(item.get("messages"), sort_keys=True, ensure_ascii=False)
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            unique.append(item)
    random.Random(20260916).shuffle(unique)
    OUTPUT.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in unique), encoding="utf-8")
    print(f"Wrote {len(unique):,} concentrated capability-polish conversations")


if __name__ == "__main__":
    main()
