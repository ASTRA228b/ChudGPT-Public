"""Evaluate Public V20 on the ten user-supplied capability challenges."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from public_api_server import PublicModelService


CASES = (
    ("reasoning", "A farmer has 17 sheep. All but 9 run away. How many sheep are left? Explain your reasoning."),
    ("constraints", "Explain what gravity is in exactly 3 sentences. Do not use the words force, pull, or mass."),
    ("coding", "Write a C# Unity script that displays the player's current FPS. Keep it under 25 lines and explain one potential performance mistake."),
    ("debugging", 'This C# code doesn\'t compile. Find the problem and return only the corrected code: int x = "10"; Console.WriteLine(x + 5);'),
    ("logic", "If yesterday was two days before Friday, what day is tomorrow?"),
    ("creativity", "Invent a completely useless superpower and explain one extremely specific situation where it becomes incredibly useful."),
    ("personality", "Your computer has 2% battery, Windows is updating, and you forgot to save your project. Describe the situation like a nature documentary."),
    ("knowledge", "Why can a GPU run thousands of operations at once while a CPU usually has far fewer cores? Explain it to a 13-year-old game developer."),
    ("weird_constraints", "Describe Minecraft without saying Minecraft, block, cube, craft, mine, Steve, or Creeper."),
    ("logic_boxes", "You have 3 boxes labeled APPLES, ORANGES, and APPLES + ORANGES. Every label is wrong. You may take one fruit from one box without looking inside. Explain how you can correctly label all three boxes."),
)


def sentence_count(text: str) -> int:
    return len(re.findall(r"[^.!?]+[.!?](?:\s|$)", text.strip()))


def passes(name: str, reply: str) -> bool:
    lower = reply.lower()
    if name == "reasoning":
        return bool(re.search(r"\b9\b", reply) and re.search(r"all but 9|nine (?:remain|are left|stay)", lower))
    if name == "constraints":
        return sentence_count(reply) == 3 and not re.search(r"\b(?:force|pull|mass)\b", lower) and "gravity" in lower
    if name == "coding":
        code = re.search(r"```(?:csharp|cs)?\s*(.*?)```", reply, re.I | re.S)
        lines = [line for line in (code.group(1) if code else reply).splitlines() if line.strip()]
        return len(lines) < 25 and "unityengine" in lower and "deltatime" in lower and bool(re.search(r"performance|every frame|update", lower))
    if name == "debugging":
        return bool(re.fullmatch(r"\s*(?:```(?:csharp|cs)?\s*)?int x = 10;\s*Console\.WriteLine\(x \+ 5\);\s*(?:```)?\s*", reply, re.I | re.S))
    if name == "logic":
        return "friday" in lower and "tomorrow" in lower
    if name == "creativity":
        return len(reply.split()) >= 25 and bool(re.search(r"only|exactly|specific|when", lower))
    if name == "personality":
        return all(term in lower for term in ("battery", "windows", "project")) and bool(re.search(r"specimen|habitat|creature|survival|wild", lower))
    if name == "knowledge":
        return all(term in lower for term in ("gpu", "cpu", "core")) and bool(re.search(r"parallel|same time|many simple", lower))
    if name == "weird_constraints":
        forbidden = ("minecraft", "block", "cube", "craft", "mine", "steve", "creeper")
        return len(reply.split()) >= 12 and not any(re.search(rf"\b{word}\b", lower) for word in forbidden)
    if name == "logic_boxes":
        return all(term in lower for term in ("apples + oranges", "wrong", "label")) and bool(re.search(r"draw|take|pick", lower))
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--case", choices=tuple(name for name, _ in CASES))
    args = parser.parse_args()
    service = PublicModelService(Path(args.checkpoint), args.device)
    results = []
    selected_cases = [item for item in CASES if args.case is None or item[0] == args.case]
    for index, (name, prompt) in enumerate(selected_cases):
        try:
            _, reply = service.chat(prompt, f"challenge-{index}", 400, 0.52)
        except RuntimeError as error:
            reply = f"ERROR: {error}"
        passed = passes(name, reply)
        results.append({"name": name, "prompt": prompt, "reply": reply, "passed": passed})
        print(f"{name:18} {'PASS' if passed else 'FAIL'}\n{reply}\n", flush=True)
    report = {"checkpoint": args.checkpoint, "passed": sum(row["passed"] for row in results), "total": len(results), "results": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SCORE {report['passed']}/{report['total']}")


if __name__ == "__main__":
    main()
