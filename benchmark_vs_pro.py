"""Compare Public and Pro through their real local HTTP APIs."""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
import uuid
import time
from pathlib import Path

PUBLIC_URL = "http://127.0.0.1:8010/api/chat"
LEGACY_URL = "http://127.0.0.1:8004/api/chat"

CASES = [
    ("greeting", ["Hello!"], ["hey|hello|hi"]),
    ("addition", ["What is 17 + 25?"], [r"\b42\b"]),
    ("subtraction", ["Calculate 50 - 19."], [r"\b31\b"]),
    ("multiplication", ["What is 12 * 8?"], [r"\b96\b"]),
    ("division", ["What is 144 / 12?"], [r"\b12\b"]),
    ("gravity", ["Explain gravity in one sentence."], ["mass"]),
    ("ocean", ["What is the largest ocean?"], ["pacific"]),
    ("sky", ["What color is the sky on a clear day?"], ["blue"]),
    ("capital", ["What is the capital of France?"], ["paris"]),
    ("planet", ["What planet is closest to the Sun?"], ["mercury"]),
    ("identity", ["What is your name?"], ["chudgpt"]),
    ("internet", ["Do you have live internet access?"], ["no"]),
    ("memory_color", ["My favorite color is teal.", "What is my favorite color?"], ["teal"]),
    ("memory_project", ["My project is Star Finch.", "What is my project?"], ["star finch"]),
    ("python_code", ["Write a Python function that adds two numbers. Code only."], ["def ", "return"]),
    ("unity_physics", ["Should Rigidbody movement use Update or FixedUpdate?"], ["fixedupdate"]),
    ("yes_no", ["Answer yes or no: Is Earth a planet?"], [r"^yes"]),
    ("short_answer", ["Short answer: What gas do humans breathe to survive?"], ["oxygen"]),
    ("correction", ["You said 4 + 4 is 9. Correct yourself."], ["8"]),
    ("topic_music", ["Let's talk about electronic music."], ["music|electronic"]),
]


def request(url: str, session: str, message: str, mode: str | None) -> str:
    payload: dict[str, object] = {"session_id": session, "message": message}
    if mode: payload["mode"] = mode
    headers = {"Content-Type": "application/json"}
    if mode:
        headers["cf-connecting-ip"] = f"benchmark-{mode}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    for attempt in range(3):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=90).read())["reply"]
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 2:
                raise
            print("Rate limited; waiting 62 seconds before retrying...")
            time.sleep(62)
    raise RuntimeError("unreachable")


def run(name: str, url: str, mode: str | None) -> dict[str, object]:
    rows = []
    for case_name, prompts, patterns in CASES:
        session = f"bench-{name}-{uuid.uuid4().hex[:12]}"
        reply = ""
        for prompt in prompts: reply = request(url, session, prompt, mode)
        passed = all(re.search(pattern, reply, re.I) for pattern in patterns)
        rows.append({"case": case_name, "passed": passed, "reply": reply})
        print(f"{name:6} {case_name:18} {'PASS' if passed else 'FAIL'} {reply[:120]!r}")
    return {"name": name, "passed": sum(row["passed"] for row in rows), "total": len(rows), "results": rows}


def main() -> None:
    report = {"public": run("Public", PUBLIC_URL, None)}
    for mode in ("buggy", "ultimate", "plus", "pro", "code", "mega"):
        report[mode] = run(mode.title(), LEGACY_URL, mode)
    output = Path("reports/benchmark_vs_pro.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL SCORES")
    for name, result in report.items():
        print(f"{name.title():10} {result['passed']}/{result['total']}")


if __name__ == "__main__": main()
