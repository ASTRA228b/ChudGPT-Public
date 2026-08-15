"""Strict, exact arithmetic for ChudGPT-Public.

Only an entire, direct two-integer arithmetic request is accepted. Everything
else returns ``None`` and follows the existing neural conversation path.
"""

from __future__ import annotations

import re
from fractions import Fraction

_DIRECT_ARITHMETIC = re.compile(
    r"^\s*(?:(?:what(?:\s+is|'s)|calculate|compute|solve|evaluate)\s+|"
    r"(?:give\s+(?:me\s+)?(?:the\s+)?(?:result|answer)(?:\s+(?:for|of))?\s+))?"
    r"([+-]?\d[\d,]*)\s*"
    r"(\+|-|\*|×|x|/|÷|plus|minus|times|multiplied\s+by|divided\s+by)\s*"
    r"([+-]?\d[\d,]*)\s*(?:\?|please|thanks)?\s*$",
    re.IGNORECASE,
)


def _exact_fraction_text(value: Fraction) -> str:
    """Render integers/terminating decimals exactly; otherwise use a fraction."""
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"
    places = max(twos, fives)
    scaled = abs(value.numerator) * (10**places) // value.denominator
    digits = str(scaled).rjust(places + 1, "0")
    rendered = f"{digits[:-places]}.{digits[-places:]}".rstrip("0").rstrip(".")
    return f"-{rendered}" if value.numerator < 0 else rendered


def exact_integer_arithmetic(message: str) -> str | None:
    """Evaluate one direct integer operation without floats or ``eval``."""
    match = _DIRECT_ARITHMETIC.fullmatch(message)
    if match is None:
        return None
    left = int(match.group(1).replace(",", ""))
    right = int(match.group(3).replace(",", ""))
    operator = re.sub(r"\s+", " ", match.group(2).lower())
    if operator in {"+", "plus"}:
        value = str(left + right)
        symbol = "+"
    elif operator in {"-", "minus"}:
        value = str(left - right)
        symbol = "-"
    elif operator in {"*", "×", "x", "times", "multiplied by"}:
        value = str(left * right)
        symbol = "*"
    else:
        symbol = "/"
        if right == 0:
            return "Division by zero is undefined."
        value = _exact_fraction_text(Fraction(left, right))
    return f"{left} {symbol} {right} = {value}"
