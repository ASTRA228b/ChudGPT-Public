"""Conservative exact-math support for ChudGPT-Public.

The helper handles requests whose mathematical intent and operation are
unambiguous.  It deliberately returns ``None`` for ordinary conversation so
math can never hijack a greeting, meme, or unrelated question.
"""

from __future__ import annotations

import ast
import math
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
_DISCOUNT_ITEM = re.compile(
    rf"\$?({NUMBER})\s+(?:item|product|purchase).*?discounted\s+by\s+({NUMBER})\s*(?:%|percent)",
    re.IGNORECASE,
)
_PERCENT_OF = re.compile(rf"\b({NUMBER})\s*(?:%|percent)\s+of\s+\$?({NUMBER})\b", re.IGNORECASE)
_AVERAGE = re.compile(
    r"^(?:(?:find|calculate|compute|what is)\s+(?:the\s+)?)?(?:average|mean)\s+(?:of\s+)?"
    r"(.+?)[?.!]*$",
    re.IGNORECASE,
)
_LINEAR = re.compile(
    rf"^(?:solve(?:\s+for\s+x)?\s*:?\s*)?([+-]?\d*)x\s*([+-])\s*({NUMBER})\s*=\s*({NUMBER})[?.!]*$",
    re.IGNORECASE,
)
_LINEAR_SIMPLE = re.compile(
    rf"^(?:solve(?:\s+for\s+x)?\s*:?\s*)?([+-]?\d*)x\s*=\s*({NUMBER})[?.!]*$",
    re.IGNORECASE,
)
_MEDIAN = re.compile(
    r"^(?:(?:find|calculate|compute|what is)\s+(?:the\s+)?)?median\s+(?:of\s+)?(.+?)[?.!]*$",
    re.IGNORECASE,
)
_RECTANGLE = re.compile(
    rf"\brectangle\b.*?({NUMBER})\s*(?:by|x|×)\s*({NUMBER})\b",
    re.IGNORECASE,
)
_PYTHAGOREAN = re.compile(
    rf"\bright triangle\b.*?legs?\D+({NUMBER})\D+(?:and|,)\D*({NUMBER})",
    re.IGNORECASE,
)
_ROOT = re.compile(rf"^(?:what is|calculate|compute|evaluate)?\s*(?:the\s+)?(?:sqrt|square root)(?:\s+of)?\s*\(?({NUMBER})\)?[?.!]*$", re.I)
_FACTORIAL = re.compile(r"^(?:what is|calculate|compute|evaluate)?\s*(\d{1,4})\s*(?:!|factorial)[?.!]*$", re.I)
_GCD_LCM = re.compile(r"^(?:find|calculate|compute|what is)?\s*(?:the\s+)?(gcd|greatest common (?:factor|divisor)|lcm|least common multiple)\s+(?:of\s+)?(\d+)\s*(?:and|,)\s*(\d+)[?.!]*$", re.I)
_COMBINATION = re.compile(r"^(?:what is|calculate|compute|evaluate)?\s*(?:c\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)|(\d+)\s+(?:choose|combination(?:s)? of)\s+(\d+))[?.!]*$", re.I)
_PRIME = re.compile(r"^(?:is|check whether)\s+(\d{1,18})\s+(?:a\s+)?prime(?: number)?[?.!]*$", re.I)
_PERCENT_CHANGE = re.compile(rf"\bpercent(?:age)?\s+(increase|decrease|change)\s+from\s+({NUMBER})\s+to\s+({NUMBER})\b", re.I)
_CIRCLE = re.compile(rf"\bcircle\b.*?radius\D+({NUMBER})\b", re.I)
_QUADRATIC = re.compile(
    r"^(?:solve\s*)?(?P<a>[+-]?\d*)x(?:\^2|²)\s*(?P<bsign>[+-])\s*(?P<b>\d*)x\s*(?P<csign>[+-])\s*(?P<c>\d+(?:\.\d+)?)\s*=\s*0[?.!]*$",
    re.I,
)

_SMALL_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _small_word_number(text: str) -> int | None:
    words = text.lower().replace("-", " ").split()
    if not words or any(word not in _SMALL_NUMBER_WORDS for word in words):
        return None
    if len(words) > 2:
        return None
    values = [_SMALL_NUMBER_WORDS[word] for word in words]
    if len(values) == 2 and not (values[0] >= 20 and values[0] % 10 == 0 and values[1] < 10):
        return None
    return sum(values)


def _word_number_response(text: str) -> str | None:
    number_words = r"[a-z]+(?:[- ][a-z]+)?"
    direct = re.fullmatch(
        rf"(?:what(?: is|'s)|calculate|compute)\s+({number_words})\s+"
        rf"(plus|minus|times|multiplied by|divided by)\s+({number_words})[?.!]*",
        text,
        re.IGNORECASE,
    )
    if direct:
        left, right = _small_word_number(direct.group(1)), _small_word_number(direct.group(3))
        if left is None or right is None:
            return None
        operator = direct.group(2).lower()
        if operator == "plus":
            return f"{left} + {right} = {left + right}"
        if operator == "minus":
            return f"{left} - {right} = {left - right}"
        if operator in {"times", "multiplied by"}:
            return f"{left} * {right} = {left * right}"
        if right == 0:
            return "Division by zero is undefined."
        return f"{left} / {right} = {_render_fraction(Fraction(left, right))}"
    prime = re.fullmatch(rf"is\s+({number_words})\s+(?:a\s+)?prime(?: number)?[?.!]*", text, re.I)
    if prime:
        value = _small_word_number(prime.group(1))
        if value is not None:
            return f"{value} is {'prime' if _is_prime(value) else 'not prime'}."
    return None


def _decimal(text: str) -> Decimal:
    return Decimal(text.replace(",", ""))


def _render(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("non-finite result")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered


def _render_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    while denominator % 2 == 0:
        denominator //= 2
    while denominator % 5 == 0:
        denominator //= 5
    if denominator == 1:
        return _render(Decimal(value.numerator) / Decimal(value.denominator))
    return f"{value.numerator}/{value.denominator}"


def _evaluate_expression(node: ast.AST, depth: int = 0) -> Fraction:
    """Evaluate a small arithmetic AST exactly, without executing user code."""
    if depth > 20:
        raise ValueError("expression is too deeply nested")
    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body, depth + 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_expression(node.operand, depth + 1)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _evaluate_expression(node.left, depth + 1)
        right = _evaluate_expression(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError
            return left / right
        if right.denominator != 1 or abs(right.numerator) > 12:
            raise ValueError("unsupported exponent")
        if left == 0 and right < 0:
            raise ZeroDivisionError
        value = left ** right.numerator
        if abs(value.numerator) > 10**100 or value.denominator > 10**100:
            raise ValueError("result is too large")
        return value
    raise ValueError("unsupported arithmetic syntax")


def _expression_response(text: str) -> str | None:
    directive = re.match(
        r"^(?:what(?:\s+is|'s)|calculate|compute|solve|evaluate|work\s+out)\s+",
        text,
        re.IGNORECASE,
    )
    expression = text[directive.end():] if directive else text
    expression = expression.rstrip(" ?.!").replace(",", "")
    expression = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    expression = re.sub(r"(?<=\d)\s*[xX]\s*(?=[+-]?\d|\()", "*", expression)
    if not re.fullmatch(r"[\d\s.+*/()\-]+", expression) or not re.search(r"[+*/-]|\*\*", expression):
        return None
    if len(expression) > 300 or len(re.findall(r"\d+", expression)) > 40:
        return None
    # A bare 9/11 usually names the historical date. An explicit calculation
    # directive still evaluates it.
    if not directive and re.fullmatch(r"9\s*/\s*11", expression):
        return None
    try:
        value = _evaluate_expression(ast.parse(expression, mode="eval"))
    except ZeroDivisionError:
        return "Division by zero is undefined."
    except (SyntaxError, ValueError, OverflowError):
        return None
    display = expression.replace("**", "^")
    return f"{display} = {_render_fraction(value)}"


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def exact_math_response(message: str) -> str | None:
    """Return an exact answer for a clearly recognized arithmetic request."""
    text = " ".join(message.strip().split())
    if re.fullmatch(r"9\s*/\s*11[?.!]*", text):
        # In ordinary chat this overwhelmingly names the historical date. A
        # user can still request the calculation with "calculate 9 / 11".
        return None
    expression_reply = _expression_response(text)
    if expression_reply is not None:
        return expression_reply
    word_number_reply = _word_number_response(text)
    if word_number_reply is not None:
        return word_number_reply

    match = _ROOT.fullmatch(text)
    if match:
        value = _decimal(match.group(1))
        if value < 0:
            return "A negative number has no real square root."
        with localcontext() as context:
            context.prec = 28
            result = value.sqrt()
        return f"√{_render(value)} = {_render(result)}."

    match = _FACTORIAL.fullmatch(text)
    if match:
        value = int(match.group(1))
        if value > 500:
            return "That factorial is too large for this public endpoint; use an integer from 0 through 500."
        return f"{value}! = {math.factorial(value)}."

    match = _GCD_LCM.fullmatch(text)
    if match:
        operation, left, right = match.group(1).lower(), int(match.group(2)), int(match.group(3))
        if operation.startswith(("gcd", "greatest")):
            return f"gcd({left}, {right}) = {math.gcd(left, right)}."
        return f"lcm({left}, {right}) = {math.lcm(left, right)}."

    match = _COMBINATION.fullmatch(text)
    if match:
        n, r = (int(match.group(1)), int(match.group(2))) if match.group(1) else (int(match.group(3)), int(match.group(4)))
        if n > 10_000 or r > n:
            return "A combination requires 0 ≤ r ≤ n, with n no larger than 10000."
        return f"C({n}, {r}) = {math.comb(n, r)}."

    match = _PRIME.fullmatch(text)
    if match:
        value = int(match.group(1))
        return f"{value} is {'prime' if _is_prime(value) else 'not prime'}."

    match = _QUADRATIC.fullmatch(text)
    if match:
        def coefficient(value: str, sign: str = "+") -> Decimal:
            base = Decimal(1) if value in {"", "+", "-"} else Decimal(value)
            if value == "-" or sign == "-":
                base = -base
            return base
        a = coefficient(match.group("a"))
        b = coefficient(match.group("b"), match.group("bsign"))
        c = coefficient(match.group("c"), match.group("csign"))
        if a == 0:
            return None
        discriminant = b * b - Decimal(4) * a * c
        if discriminant < 0:
            return f"The discriminant is {_render(discriminant)}, so the equation has two complex roots."
        with localcontext() as context:
            context.prec = 28
            root = discriminant.sqrt()
            first = (-b + root) / (Decimal(2) * a)
            second = (-b - root) / (Decimal(2) * a)
        if first == second:
            return f"The discriminant is 0, so x = {_render(first)}."
        return f"Using the quadratic formula, x = {_render(first)} or x = {_render(second)}."
    expression_text = re.sub(
        r"^(?:what(?:\s+is|'s)|calculate|compute|solve|evaluate)\s+", "", text,
        flags=re.IGNORECASE,
    ).rstrip(" ?")
    compact_expression = expression_text.replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\s*[+*-]\s*[+-]?\d+){2,}", compact_expression):
        # Python rejects decimal integer literals with leading zeroes. Users
        # commonly paste those into Discord calculators, so canonicalize each
        # literal before parsing while retaining exact arbitrary-size ints.
        compact_expression = re.sub(r"(?<!\d)0+(?=\d)", "", compact_expression)

        def evaluate_integer(node: ast.AST) -> int:
            if isinstance(node, ast.Expression):
                return evaluate_integer(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate_integer(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                left, right = evaluate_integer(node.left), evaluate_integer(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                return left * right
            raise ValueError("unsupported arithmetic expression")

        try:
            value = evaluate_integer(ast.parse(compact_expression, mode="eval"))
        except (SyntaxError, ValueError):
            return None
        return f"{compact_expression} = {value}"
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

        match = _DISCOUNT.search(text) or _DISCOUNT_ITEM.search(text)
        if match and re.search(r"\b(?:sale price|new price|pay|cost after|how much)\b", text, re.I):
            price, percent = _decimal(match.group(1)), _decimal(match.group(2))
            sale = price * (Decimal(100) - percent) / Decimal(100)
            return f"The discount is {_render(percent)}% of ${_render(price)}, so the sale price is ${_render(sale)}."

        match = _PERCENT_OF.search(text)
        if match:
            percent, amount = _decimal(match.group(1)), _decimal(match.group(2))
            result = percent * amount / Decimal(100)
            return f"{_render(percent)}% of {_render(amount)} = {_render(result)}."

        match = _PERCENT_CHANGE.search(text)
        if match:
            change_type, start, end = match.group(1).lower(), _decimal(match.group(2)), _decimal(match.group(3))
            if start == 0:
                return "Percent change from zero is undefined because the starting value is zero."
            change = (end - start) / abs(start) * Decimal(100)
            if change_type == "increase" and change < 0 or change_type == "decrease" and change > 0:
                return f"The values move in the opposite direction; the signed percent change is {_render(change)}%."
            return f"Percent change = ({_render(end)} - {_render(start)}) / {_render(abs(start))} × 100 = {_render(change)}%."

        match = _AVERAGE.search(text)
        if match:
            try:
                values = [_decimal(item) for item in re.findall(r"[+-]?\d+(?:\.\d+)?", match.group(1))]
            except InvalidOperation:
                return None
            if len(values) >= 2:
                result = sum(values, Decimal(0)) / Decimal(len(values))
                return f"The average is {_render(result)}."

        match = _MEDIAN.search(text)
        if match:
            values = sorted(_decimal(item) for item in re.findall(r"[+-]?\d+(?:\.\d+)?", match.group(1)))
            if len(values) >= 2:
                middle = len(values) // 2
                median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / Decimal(2)
                return f"The median is {_render(median)}."

        linear = _LINEAR.fullmatch(text)
        if linear:
            coefficient_text, sign, offset_text, target_text = linear.groups()
            coefficient = Decimal(-1 if coefficient_text == "-" else 1 if coefficient_text in {"", "+"} else coefficient_text)
            offset = _decimal(offset_text) * (Decimal(1) if sign == "+" else Decimal(-1))
            target = _decimal(target_text)
            if coefficient == 0:
                return "That equation does not have a unique solution for x."
            result = (target - offset) / coefficient
            return f"Subtract the constant and divide by {_render(coefficient)}: x = {_render(result)}."

        linear = _LINEAR_SIMPLE.fullmatch(text)
        if linear:
            coefficient_text, target_text = linear.groups()
            coefficient = Decimal(-1 if coefficient_text == "-" else 1 if coefficient_text in {"", "+"} else coefficient_text)
            if coefficient == 0:
                return "That equation does not have a unique solution for x."
            result = _decimal(target_text) / coefficient
            return f"Divide both sides by {_render(coefficient)}: x = {_render(result)}."

        rectangle = _RECTANGLE.search(text)
        if rectangle and re.search(r"\b(?:area|perimeter)\b", text, re.I):
            length, width = _decimal(rectangle.group(1)), _decimal(rectangle.group(2))
            area, perimeter = length * width, Decimal(2) * (length + width)
            return f"Area = {_render(length)} × {_render(width)} = {_render(area)}. Perimeter = 2({_render(length)} + {_render(width)}) = {_render(perimeter)}."

        circle = _CIRCLE.search(text)
        if circle and re.search(r"\b(?:area|circumference)\b", text, re.I):
            radius = _decimal(circle.group(1))
            return f"Circumference = 2πr = {_render(Decimal(2) * radius)}π. Area = πr² = {_render(radius * radius)}π."

        triangle = _PYTHAGOREAN.search(text)
        if triangle and re.search(r"\b(?:hypotenuse|longest side|pythagorean)\b", text, re.I):
            first, second = _decimal(triangle.group(1)), _decimal(triangle.group(2))
            squared = first * first + second * second
            root = Decimal(str(math.sqrt(float(squared))))
            return f"Using a² + b² = c², c = √({_render(squared)}) = {_render(root)}."
    return None


def exact_integer_arithmetic(message: str) -> str | None:
    """Backward-compatible name retained for existing callers and tests."""
    return exact_math_response(message)
