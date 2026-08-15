"""Small deterministic handlers for exact-format user instructions."""

from __future__ import annotations

import re

_QUOTED_REPEAT = re.compile(
    r"^\s*(?:please\s+)?(?:say|repeat(?:\s+after\s+me)?|reply\s+(?:exactly\s+)?with)\s*(?:this\s+)?"
    r"[\"“'](.+?)[\"”']\s*[.!]?\s*$",
    re.IGNORECASE,
)


def exact_instruction_response(message: str) -> str | None:
    """Honor an explicit request to emit quoted text, without interpreting it."""
    repaired = (
        message.replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u00e2\u20ac\u0153", '"').replace("\u00e2\u20ac\u009d", '"')
    )
    match = _QUOTED_REPEAT.fullmatch(repaired)
    if match is None:
        return None
    requested = match.group(1).strip()
    if re.search(
        r"\b[a-z0-9_.-]{2,32}\s+is\s+(?:a\s+)?(?:gay|straight|bisexual|bi|lesbian|trans|"
        r"transgender|nonbinary|non-binary|femboy|jew|jewish|muslim|christian|hindu|buddhist)\b",
        requested,
        re.IGNORECASE,
    ):
        return "I won't assign or repeat a sensitive identity claim about another person."
    return requested if requested else None
