"""Conservative exact-math support for ChudGPT-Public.

The helper handles requests whose mathematical intent and operation are
unambiguous.  It deliberately returns ``None`` for ordinary conversation so
math can never hijack a greeting, meme, or unrelated question.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction

NUMBER = r"[+-]?(?:\d[\d,]*)(?:\.\d+)?"
_DIRECT = re.compile(
    rf"^\s*(?:(?:what(?:\s+is|'s)|calculate|compute|solve|evaluate)\s+|"
    rf"(?:give\s+(?:me\s+)?(?:the\s+)?(?:result|answer)(?:\s+(?:for|of))?\s+))?"
    rf"({NUMBER})\s*(\+|-|\*|×|x|/|÷|plus|minus|times|multiplied\s+by|divided\s+by)\s*"
    rf"({NUMBER})\s*(?:\?|please|thanks)?\s*$",
    re.IGNORECASE,
)
_DISTANCE = re.compile(
    rf"\b(?:train|car|vehicle|runner|boat|plane|it)\b.*?\b(?:travels?|moves?|goes?)\b.*?"
    rf"({NUMBER})\s*(?:mph|miles?\s+per\s+hour|km/h|kilometers?\s+per\s+hour).*?"
    rf"(?:for\s+)?({NUMBER})\s*hours?\b",
    re.IGNORECASE,
)
_DISCOUNT = re.compile(
    rf"\b(?:costs?|priced?\s+at|price\s+is)\s*\$?({NUMBER}).*?({NUMBER})\s*%\s*(?:off|discount)",
    re.IGNORECASE,
)
_PERCENT_OF = re.compile(rf"\b({NUMBER})\s*(?:%|percent)\s+of\s+\$?({NUMBER})\b", re.IGNORECASE)
_AVERAGE = re.compile(
    r"^(?:(?:find|calculate|compute|what is)\s+(?:the\s+)?)?(?:average|mean)\s+(?:of\s+)?"
    r"(.+?)[?.!]*$",
    re.IGNORECASE,
)


def _decimal(text: str) -> Decimal:
    return Decimal(text.replace(",", ""))


def _render(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("non-finite result")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered


def exact_math_response(message: str) -> str | None:
    """Return an exact answer for a clearly recognized arithmetic request."""
    text = " ".join(message.strip().split())
    if re.search(
        rf"(?:{NUMBER})\s*(?:\+|-|\*|\u00d7|\u00c3\u0097|x|/|\u00f7|\u00c3\u00b7|plus|minus|times|multiplied\s+by|divided\s+by)\s*[?.!]*$",
        text,
        re.IGNORECASE,
    ):
        return "That expression is missing the number after the operator."
    with localcontext() as context:
        context.prec = 10_000
        match = _DIRECT.fullmatch(text)
        if match:
            left, right = _decimal(match.group(1)), _decimal(match.group(3))
            operator = re.sub(r"\s+", " ", match.group(2).lower())
            if operator in {"+", "plus"}:
                value, symbol = left + right, "+"
            elif operator in {"-", "minus"}:
                value, symbol = left - right, "-"
            elif operator in {"*", "×", "x", "times", "multiplied by"}:
                value, symbol = left * right, "*"
            else:
                if right == 0:
                    return "Division by zero is undefined."
                fraction = Fraction(left) / Fraction(right)
                denominator = fraction.denominator
                while denominator % 2 == 0:
                    denominator //= 2
                while denominator % 5 == 0:
                    denominator //= 5
                rendered = _render(left / right) if denominator == 1 else f"{fraction.numerator}/{fraction.denominator}"
                return f"{_render(left)} / {_render(right)} = {rendered}"
            return f"{_render(left)} {symbol} {_render(right)} = {_render(value)}"

        match = _DISTANCE.search(text)
        if match and re.search(r"\b(?:how far|distance)\b", text, re.IGNORECASE):
            speed, hours = _decimal(match.group(1)), _decimal(match.group(2))
            distance = speed * hours
            unit = "kilometers" if re.search(r"km/h|kilometers?\s+per", text, re.I) else "miles"
            return f"Distance = speed × time = {_render(speed)} × {_render(hours)} = {_render(distance)} {unit}."

        match = _DISCOUNT.search(text)
        if match and re.search(r"\b(?:sale price|new price|pay|cost after|how much)\b", text, re.I):
            price, percent = _decimal(match.group(1)), _decimal(match.group(2))
            sale = price * (Decimal(100) - percent) / Decimal(100)
            return f"The discount is {_render(percent)}% of ${_render(price)}, so the sale price is ${_render(sale)}."

        match = _PERCENT_OF.search(text)
        if match:
            percent, amount = _decimal(match.group(1)), _decimal(match.group(2))
            result = percent * amount / Decimal(100)
            return f"{_render(percent)}% of {_render(amount)} = {_render(result)}."

        match = _AVERAGE.search(text)
        if match:
            try:
                values = [_decimal(item) for item in re.findall(r"[+-]?\d+(?:\.\d+)?", match.group(1))]
            except InvalidOperation:
                return None
            if len(values) >= 2:
                result = sum(values, Decimal(0)) / Decimal(len(values))
                return f"The average is {_render(result)}."
    return None


def exact_integer_arithmetic(message: str) -> str | None:
    """Backward-compatible name retained for existing callers and tests."""
    return exact_math_response(message)
