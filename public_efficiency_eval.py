"""Held-out Public tests for intent precision and useful information per token."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class EfficiencyCase:
    category: str
    prompts: tuple[str, ...]
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    ideal_max_tokens: int = 45


CASES = (
    EfficiencyCase("math_false_positive", ("No math",), required=("no math|other topic|talk",), forbidden=(r"\banswer is\b", r"\d+\s*[+*/=-]")),
    EfficiencyCase("math_false_positive", ("Nothing",), required=("fine|okay|quiet|mind",), forbidden=(r"\banswer is\b", r"\d+\s*[+*/=-]")),
    EfficiencyCase("math_false_positive", ("I have 2 dogs",), forbidden=(r"\banswer is\b", r"\b2\s*[+*/=-]")),
    EfficiencyCase("math_false_positive", ("The movie 1917 was intense",), forbidden=(r"\banswer is\b", r"calculate|equals")),
    EfficiencyCase("math_false_positive", ("67",), forbidden=(r"\b67\s*[+*/=-]",)),
    EfficiencyCase("math_true_positive", ("What is 18 + 24?",), required=(r"\b42\b",)),
    EfficiencyCase("math_true_positive", ("Calculate 12.5 percent of 80.",), required=(r"\b10\b",)),
    EfficiencyCase("negation", ("Give me a coding idea.", "Don't give me code. Just name the idea."), forbidden=("```", "def ", "class ")),
    EfficiencyCase("negation", ("Explain fractions.", "Not that. Change the topic to music."), required=("music",), forbidden=("numerator", "denominator")),
    EfficiencyCase("negation", ("Tell me the full explanation.", "Stop explaining."), forbidden=(r"\bfirst\b.*\bsecond\b",)),
    EfficiencyCase("short_context", ("I chose the red backpack.", "why"), required=("red|backpack|chose",)),
    EfficiencyCase("short_context", ("Let's discuss dogs.", "how"), required=("dog",)),
    EfficiencyCase("short_context", ("That plan seems risky.", "you sure?"), required=("risk|sure|check|uncertain",)),
    EfficiencyCase("short_context", ("Want to hear what happened?", "yeah"), forbidden=("calculate", "```")),
    EfficiencyCase("short_context", ("I failed the level again.", "lol"), required=("level|game|again|rough|funny",)),
    EfficiencyCase("meme", ("bro is NOT beating the allegations",), required=("joke|accus|evidence|behavior|meme",)),
    EfficiencyCase("meme", ("average discord mod",), required=("stereotype|discord|joke|mock",)),
    EfficiencyCase("meme", ("this is peak",), required=("best|excellent|praise|good",)),
    EfficiencyCase("meme", ("what does cooked mean here: I forgot the assignment",), required=("trouble|doomed|bad|unprepared",)),
    EfficiencyCase("meme", ("He lost 500 aura points",), required=("cool|status|embarrass|joke",)),
    EfficiencyCase("meme", ("Explain the Virgin vs Chad format",), required=("contrast|insecure|confident|traits",)),
    EfficiencyCase("meme", ("What is a Wojak reaction image?",), required=("emotion|reaction|character|feeling",)),
    EfficiencyCase("meme", ("Is 'real' always a meme?",), required=("agree|context|not always",)),
)


def token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def quality_per_token(case: EfficiencyCase, reply: str) -> float:
    """Reward required information, penalize errors, then gently penalize bloat."""
    required_hits = sum(bool(re.search(pattern, reply, re.I | re.S)) for pattern in case.required)
    forbidden_hits = sum(bool(re.search(pattern, reply, re.I | re.S)) for pattern in case.forbidden)
    correctness = 1.0 if not case.required else required_hits / len(case.required)
    correctness = max(0.0, correctness - 0.5 * forbidden_hits)
    tokens = token_count(reply)
    if tokens <= case.ideal_max_tokens:
        efficiency = 1.0
    else:
        efficiency = max(0.35, case.ideal_max_tokens / max(1, tokens))
    return round(100.0 * correctness * efficiency, 2)
