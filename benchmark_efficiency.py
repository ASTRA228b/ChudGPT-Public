"""Run the held-out intent and quality-per-token suite against Public's API."""

from __future__ import annotations

import argparse
import json
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path

from public_efficiency_eval import CASES, quality_per_token, token_count


def send(url: str, session_id: str, message: str) -> str:
    body = json.dumps({"message": message, "session_id": session_id, "max_new_tokens": 120}).encode()
    request = urllib.request.Request(url.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return str(json.loads(response.read().decode())["reply"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--output", default="reports/public_efficiency.json")
    args = parser.parse_args()
    rows, category_scores = [], defaultdict(list)
    for index, case in enumerate(CASES, 1):
        session = "efficiency-" + uuid.uuid4().hex
        replies = [send(args.url, session, prompt) for prompt in case.prompts]
        score = quality_per_token(case, replies[-1])
        category_scores[case.category].append(score)
        rows.append({"category": case.category, "prompts": case.prompts, "replies": replies, "tokens": token_count(replies[-1]), "score": score})
        preview = replies[-1][:80].encode("ascii", "backslashreplace").decode("ascii")
        print(f"{index:02}/{len(CASES)} {case.category:20} {score:6.2f} {preview!r}")
    summary = {name: round(sum(scores) / len(scores), 2) for name, scores in category_scores.items()}
    summary["overall"] = round(sum(row["score"] for row in rows) / len(rows), 2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
