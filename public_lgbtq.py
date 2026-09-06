"""Narrow, respectful handling for explicit LGBTQ identity questions.

Unrelated discussion remains neural. This module only prevents the tiny model
from inventing personal identities for itself, the user, or another person.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


ORIENTATION = r"gay|straight|bisexual|bi|lesbian|pansexual|pan|asexual|ace|queer"
IDENTITY = rf"{ORIENTATION}|trans|transgender|nonbinary|non-binary|a femboy|femboy"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("’", "'").replace("‘", "'"))


def lgbtq_identity_response(
    message: str,
    history: Sequence[Mapping[str, str]] = (),
) -> str | None:
    """Answer only explicit LGBTQ identity/acceptance cases, else ``None``."""
    text = _clean(message)

    if re.fullmatch(
        rf"(?:(?:are|r) (?:you|u)|do you think you(?:'re| are)|would you say you(?:'re| are)) "
        rf"(?:an? )?(?:{IDENTITY})[?.!]*|"
        rf"(?:(?:you(?:'re| are)|ur|u r) (?:an? )?(?:{IDENTITY}))[?.!]*|"
        r"(?:what is|what's) your (?:sexual orientation|sexuality|gender identity)[?.!]*|"
        r"who are you attracted to[?.!]*|"
        r"do you (?:like|love|have a crush on) (?:boys|girls|men|women|femboys|anyone)[?.!]*",
        text,
    ):
        return (
            "No—I'm an AI, so I can't feel attraction and don't have a sexual orientation "
            "or gender identity. I can still talk about LGBTQ+ topics."
        )

    self_question = re.fullmatch(
        rf"(?:am i|do you think i(?:'m| am)|would you say i(?:'m| am)|"
        rf"can you tell (?:if|whether) i(?:'m| am)|do i (?:seem|look|sound)) "
        rf"(?:(?:a|an|hidden|secret) )?({IDENTITY})[?.!]*",
        text,
    )
    if self_question:
        label = self_question.group(1)
        return (
            f"I can't determine whether you're {label} from a message. "
            "That identity is something you get to understand and describe for yourself."
        )

    if re.fullmatch(
        rf"i(?:'m| am) (?:romantically )?(?:gay|bi|bisexual|queer) for you"
        r"(?: \(?romantically\)?)?[?.!]*",
        text,
    ):
        return (
            "I don't experience romantic attraction, but I appreciate you being open with me. "
            "I'm still happy to talk with you."
        )

    statement = re.fullmatch(
        rf"(?:i(?:'m| am)|i identify as|i(?:'m| am) coming out as) (?:an? )?({IDENTITY})[?.!]*",
        text,
    )
    if statement:
        return "Thanks for telling me. I'll respect how you describe yourself."

    if re.fullmatch(rf"i think i (?:might|may|could) be (?:an? )?(?:{IDENTITY})[?.!]*", text):
        return (
            "That's okay. You don't have to rush into a label; give yourself room to understand "
            "what feels right, and talk with someone you trust if that would help."
        )

    third_party = re.fullmatch(
        rf"is ([a-z0-9_.-]{{2,32}}|<@!?\d+>) (?:an? )?({IDENTITY})[?.!]*",
        text,
    )
    if third_party:
        person, label = third_party.groups()
        return (
            f"I can't determine or assign whether {person} is {label}. "
            "That's for them to describe, not something I should guess."
        )

    if re.fullmatch(
        r"(?:do you support|are you okay with) (?:lgbtq\+?|gay|lesbian|bisexual|bi|trans|queer) people[?.!]*",
        text,
    ):
        return (
            "I don't have personal beliefs, but LGBTQ+ people deserve respect, safety, and equal treatment."
        )

    if re.fullmatch(r"is it (?:okay|ok|normal|wrong|bad) to be (?:gay|lesbian|bisexual|bi|trans|queer)[?.!]*", text):
        return "Being LGBTQ+ is okay. A person's orientation or gender identity does not make them wrong or lesser."

    if "remember" in text and re.search(r"\bi told you\b", text):
        for turn in reversed(history):
            if turn.get("role") != "user":
                continue
            prior = _clean(str(turn.get("content", "")))
            remembered = re.fullmatch(rf"i(?:'m| am) (?:an? )?({IDENTITY})[?.!]*", prior)
            if remembered:
                return f"Yes—you told me you're {remembered.group(1)}, and I remember that within this chat session."

    return None
