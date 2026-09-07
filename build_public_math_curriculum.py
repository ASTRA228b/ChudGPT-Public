"""Build a deterministic, broad neural math curriculum for Public V20."""

from __future__ import annotations

import json
import math
import random
from fractions import Fraction
from pathlib import Path


OUTPUT = Path("data/public_v20_math_curriculum.jsonl")
TRAINING_OUTPUT = Path("data/public_v20_math_training_mix.jsonl")
SEED = 20260906


def clean_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}".rstrip("0").rstrip(".")


def add(rows: list[dict], prompt: str, answer: str, topic: str) -> None:
    rows.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], "source": f"public-v20-math-{topic}"})


def main() -> None:
    rng = random.Random(SEED)
    rows: list[dict] = []

    # Early elementary: exhaustive small-number facts in multiple phrasings.
    for left in range(0, 51):
        for right in range(0, 51):
            total = left + right
            add(rows, f"What is {left} + {right}?", f"{left} + {right} = {total}.", "addition")
            if left >= right:
                add(rows, f"Calculate {left} - {right}.", f"{left} - {right} = {left - right}.", "subtraction")
    for left in range(0, 13):
        for right in range(0, 13):
            add(rows, f"What is {left} times {right}?", f"{left} × {right} = {left * right}.", "multiplication")
            if right:
                product = left * right
                add(rows, f"Compute {product} / {right}.", f"{product} ÷ {right} = {left}.", "division")

    # Upper elementary and middle school: mixed arithmetic and word problems.
    for _ in range(900):
        left, right = rng.randint(20, 999), rng.randint(2, 99)
        op = rng.choice(("+", "-", "*"))
        value = {"+": left + right, "-": left - right, "*": left * right}[op]
        add(rows, f"Calculate {left} {op} {right}.", f"{left} {op} {right} = {value}.", "arithmetic")
    for _ in range(500):
        whole, places = rng.randint(2, 80), rng.randint(2, 15)
        total = whole * places
        add(rows, f"A teacher puts {total} pencils equally into {places} boxes. How many pencils are in each box?", f"Divide {total} by {places}: {total} ÷ {places} = {whole}. Each box has {whole} pencils.", "word-problem")
    for _ in range(500):
        a, b, c = rng.randint(1, 30), rng.randint(1, 20), rng.randint(1, 15)
        value = a + b * c
        add(rows, f"Evaluate {a} + {b} × {c} using order of operations.", f"Multiply first: {b} × {c} = {b*c}. Then add {a}: the answer is {value}.", "order-operations")

    # Fractions, decimals, ratios, and percentages.
    for _ in range(700):
        a, b, c, d = rng.randint(1, 12), rng.randint(2, 12), rng.randint(1, 12), rng.randint(2, 12)
        op = rng.choice(("+", "-", "*"))
        first, second = Fraction(a, b), Fraction(c, d)
        result = {"+": first + second, "-": first - second, "*": first * second}[op]
        result_text = str(result.numerator) if result.denominator == 1 else f"{result.numerator}/{result.denominator}"
        add(rows, f"Simplify {a}/{b} {op} {c}/{d}.", f"The simplified result is {result_text}.", "fractions")
    for _ in range(500):
        left = rng.randint(-500, 500) / 10
        right = rng.randint(-200, 200) / 10
        result = left + right
        add(rows, f"What is {clean_number(left)} + {clean_number(right)}?", f"{clean_number(left)} + {clean_number(right)} = {clean_number(result)}.", "decimals")
    for _ in range(600):
        percent = rng.choice((5, 10, 15, 20, 25, 30, 40, 50, 60, 75))
        whole = rng.randint(2, 500) * 20
        result = whole * percent / 100
        add(rows, f"Find {percent}% of {whole}.", f"{percent}% of {whole} is {clean_number(result)}.", "percent")
    for _ in range(350):
        first, second, scale = rng.randint(1, 15), rng.randint(1, 15), rng.randint(2, 20)
        add(rows, f"The ratio is {first}:{second}. If the first amount is {first*scale}, what is the second amount?", f"The scale factor is {scale}, so the second amount is {second} × {scale} = {second*scale}.", "ratios")

    # Algebra, coordinate geometry, geometry, statistics, and high-school basics.
    for _ in range(800):
        x, coefficient, offset = rng.randint(-40, 40), rng.randint(1, 12), rng.randint(-30, 30)
        target = coefficient * x + offset
        add(rows, f"Solve for x: {coefficient}x + ({offset}) = {target}.", f"Subtract {offset}, then divide by {coefficient}. x = {x}.", "linear-algebra")
    for _ in range(350):
        x1, y1 = rng.randint(-20, 20), rng.randint(-20, 20)
        dx, slope = rng.randint(1, 10), rng.randint(-8, 8)
        x2, y2 = x1 + dx, y1 + slope * dx
        add(rows, f"Find the slope through ({x1}, {y1}) and ({x2}, {y2}).", f"Slope = ({y2} - {y1}) / ({x2} - {x1}) = {slope}.", "coordinate-geometry")
    for _ in range(350):
        length, width = rng.randint(2, 50), rng.randint(2, 50)
        add(rows, f"A rectangle is {length} by {width}. Find its area and perimeter.", f"Area = {length} × {width} = {length*width}. Perimeter = 2({length} + {width}) = {2*(length+width)}.", "geometry")
    for _ in range(250):
        radius = rng.randint(1, 30)
        add(rows, f"Find the circumference and area of a circle with radius {radius}. Leave answers in terms of pi.", f"Circumference = 2πr = {2*radius}π. Area = πr² = {radius*radius}π.", "geometry")
    for _ in range(300):
        values = [rng.randint(0, 50) for _ in range(5)]
        mean = sum(values) / len(values)
        ordered = sorted(values)
        add(rows, f"Find the mean and median of {', '.join(map(str, values))}.", f"The mean is {clean_number(mean)} and the median is {ordered[2]}.", "statistics")
    for _ in range(250):
        root1, root2 = rng.randint(-12, 12), rng.randint(-12, 12)
        b, c = -(root1 + root2), root1 * root2
        add(rows, f"Solve x^2 + ({b})x + ({c}) = 0.", f"The expression factors as (x - ({root1}))(x - ({root2})) = 0, so x = {root1} or x = {root2}.", "quadratics")
    for _ in range(200):
        base, exponent = rng.randint(2, 12), rng.randint(2, 6)
        add(rows, f"Evaluate {base}^{exponent}.", f"{base}^{exponent} = {base**exponent}.", "exponents")
    for _ in range(200):
        n, r = rng.randint(5, 20), rng.randint(0, 5)
        r = min(r, n)
        combinations = math.comb(n, r)
        add(rows, f"How many ways can you choose {r} objects from {n} objects when order does not matter?", f"Use C({n}, {r}) = {n}! / ({r}!({n-r})!) = {combinations}.", "combinatorics")

    rng.shuffle(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} examples to {OUTPUT}")

    # Retain general conversation and the ten-skill challenge while math is
    # emphasized. These are training examples, never runtime answer routes.
    retention: list[str] = []
    retention.extend(Path("data/public_v20_focused.jsonl").read_text(encoding="utf-8").splitlines())
    # Acceptance benchmarks are evaluation-only.  Never mix their prompts or
    # reference answers into training; that measures memorization, not math.
    with TRAINING_OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for line in retention:
            if line.strip():
                handle.write(line.strip() + "\n")
    print(f"Wrote {len(rows) + len(retention):,} examples to {TRAINING_OUTPUT}")


if __name__ == "__main__":
    main()
