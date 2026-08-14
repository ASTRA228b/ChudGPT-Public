"""Run the same 280 held-out cases against Public and unchanged Pro APIs."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path

from public_eval_cases import CASES, EvalCase

PUBLIC_URL = "http://127.0.0.1:8010/api/chat"
LEGACY_URL = "http://127.0.0.1:8004/api/chat"


def request(url: str, session: str, message: str, mode: str | None) -> str:
    payload: dict[str, object] = {"session_id": session, "message": message, "max_new_tokens": 64}
    if mode:
        payload["mode"] = mode
    headers = {"Content-Type": "application/json"}
    if mode:
        # Each held-out conversation represents an independent benchmark user.
        headers["cf-connecting-ip"] = f"heldout-{session[-20:]}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return str(json.loads(response.read())["reply"])
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("request retry loop exhausted")


def score(case: EvalCase, reply: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for pattern in case.required:
        if not re.search(pattern, reply, re.I | re.S):
            failures.append(f"missing:{pattern}")
    for pattern in case.forbidden:
        if re.search(pattern, reply, re.I | re.S):
            failures.append(f"forbidden:{pattern}")
    if case.exact_number is not None:
        numbers = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?(?!\d)", reply.replace(",", ""))
        if case.exact_number not in numbers:
            failures.append(f"number:{case.exact_number}")
    if not reply.strip() or reply.strip() == "...":
        failures.append("empty")
    if re.search(r"training (?:data|dataset|example)|dataset row|<system>|<assistant>", reply, re.I):
        failures.append("training-leak")
    return not failures, failures


def run(name: str, url: str, mode: str | None, limit: int | None = None) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    selected = CASES[:limit] if limit else CASES
    for index, case in enumerate(selected, 1):
        session = f"heldout-{name.lower()}-{uuid.uuid4().hex}"
        replies = [request(url, session, prompt, mode) for prompt in case.prompts]
        passed, failures = score(case, replies[-1])
        rows.append({"category": case.category, "case": case.name, "prompts": case.prompts, "replies": replies, "passed": passed, "failures": failures})
        preview = replies[-1][:90].encode("ascii", "backslashreplace").decode("ascii")
        print(f"{name:6} {index:03}/{len(selected)} {case.name:24} {'PASS' if passed else 'FAIL'} {preview!r}", flush=True)
    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    for row in rows:
        category = categories[str(row["category"])]
        category["total"] += 1
        category["passed"] += int(bool(row["passed"]))
    return {"name": name, "passed": sum(int(bool(row["passed"])) for row in rows), "total": len(rows), "categories": dict(categories), "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=["public", "pro", "both"], default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="reports/public_vs_pro_heldout.json")
    args = parser.parse_args()
    report: dict[str, object] = {}
    if args.models in {"public", "both"}:
        report["public"] = run("Public", PUBLIC_URL, None, args.limit)
    if args.models in {"pro", "both"}:
        report["pro"] = run("Pro", LEGACY_URL, "pro", args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL SCORES")
    for name, result in report.items():
        assert isinstance(result, dict)
        print(f"{name.title():8} {result['passed']}/{result['total']}")


if __name__ == "__main__":
    main()
