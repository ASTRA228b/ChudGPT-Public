"""Small deterministic handlers for exact-format user instructions."""

from __future__ import annotations

import re

_QUOTED_REPEAT = re.compile(
    r"^\s*(?:please\s+)?(?:say|repeat(?:\s+after\s+me)?|reply\s+(?:exactly\s+)?with)\s*"
    r"[\"“'](.+?)[\"”']\s*[.!]?\s*$",
    re.IGNORECASE,
)


def exact_instruction_response(message: str) -> str | None:
    """Honor an explicit request to emit quoted text, without interpreting it."""
    match = _QUOTED_REPEAT.fullmatch(message)
    if match is None:
        return None
    requested = match.group(1).strip()
    return requested if requested else None
