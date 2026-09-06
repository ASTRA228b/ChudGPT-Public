"""Fast held-out benchmark for the Public V20 canned greeting system."""

from __future__ import annotations

import sys

from public_greetings import canned_greeting_response


CASES = (
    "Hi", "hello", "Hey there!", "yo", "Hello ChudGPT", "Hey mate",
    "Good morning", "Good afternoon!", "Good evening",
    "How are you?", "hru", "How are you doing?", "How's it going?",
    "Hey mate, how is it going?", "What's up?", "sup", "wassup",
    "hola", "bonjour", "こんにちは", "你好", "안녕하세요", "مرحبا",
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    passed = 0
    for prompt in CASES:
        reply = canned_greeting_response(prompt)
        ok = bool(reply)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} {prompt!r} -> {reply!r}")
    print(f"SCORE {passed}/{len(CASES)}")
    raise SystemExit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
