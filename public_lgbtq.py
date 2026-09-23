"""Respectful handling for explicit LGBTQIA+ questions and disclosures.

Unrelated discussion remains neural. This module only prevents the tiny model
from inventing identities and restores basic awareness independently of weights.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import emoji

from chudlm.emoji_awareness import strip_emoji_context


ORIENTATION = r"gay|straight|bisexual|bi|lesbian|pansexual|pan|asexual|ace|queer"
IDENTITY = rf"{ORIENTATION}|trans|transgender|nonbinary|non-binary|intersex|aromantic|agender|genderfluid|a femboy|femboy"
COMMUNITY = r"lgbt(?:q(?:ia)?)?\+?"
PRONOUN_STATEMENT = re.compile(r"my pronouns are ([a-z]+(?:\s*/\s*[a-z]+){1,2})[.!]*")
DEFINITIONS = {
    "gay": "Gay usually describes attraction to people of the same gender. People choose their own labels.",
    "lesbian": "Lesbian commonly describes women attracted to women; some nonbinary people also use this label.",
    "bisexual": "Bisexual means attraction to more than one gender; it doesn't have to be equal or happen at the same time.",
    "pansexual": "Pansexual describes attraction to people regardless of gender.",
    "asexual": "Asexual describes experiencing little or no sexual attraction. Asexual people can still have romantic relationships.",
    "aromantic": "Aromantic describes experiencing little or no romantic attraction. It is distinct from asexuality.",
    "transgender": "Transgender describes a gender identity different from the sex someone was assigned at birth. Gender identity and sexual orientation are different things.",
    "nonbinary": "Nonbinary describes gender identities that aren't exclusively man or woman. Use the name and pronouns the person asks for.",
    "intersex": "Intersex describes variations in sex characteristics that don't fit typical definitions of male or female. It doesn't determine someone's gender or orientation.",
    "queer": "Queer is an umbrella label some LGBTQIA+ people use. It has also been used as a slur, so don't apply it to someone who doesn't use it for themselves.",
}


def _clean(text: str) -> str:
    text = strip_emoji_context(text)
    text = re.sub(r"\s+", " ", text.strip().lower().replace("’", "'").replace("‘", "'"))
    text = re.sub(r"^im\b", "i'm", text)
    # Decorations may accompany a disclosure; never discard substantive words.
    text = re.sub(r"\s*~?uwu~?\s*$", "", text)
    text = re.sub(r"(?:\s*[^\w\s?.!]+)+$", lambda m: "" if emoji.emoji_list(m[0]) else m[0], text)
    return text.strip()


def lgbtq_identity_response(
    message: str,
    history: Sequence[Mapping[str, str]] = (),
    *,
    factual_only: bool = False,
) -> str | None:
    """Answer only explicit LGBTQ identity/acceptance cases, else ``None``."""
    text = _clean(message)
    # The live chat uses this module for explicit facts and recalled pronouns,
    # not acknowledgments or the model's conversational persona.
    if factual_only and re.search(rf"\b(?:{IDENTITY})\b|sexuality|sexual orientation|gender identity|attracted to|crush on", text):
        if re.match(r"(?:i(?:'m| am| identify| think)|(?:are|r) (?:you|u)|you|ur\b|your\b|u r\b|do you (?:like|love|think)|would you say you|who are you attracted|what(?: is|'s) your)", text):
            return None

    if re.fullmatch(r"(?:what are my pronouns|do you remember my pronouns)[?.!]*", text):
        for turn in reversed(history):
            if turn.get("role") == "user":
                found = PRONOUN_STATEMENT.fullmatch(_clean(str(turn.get("content", ""))))
                if found:
                    return f"You told me your pronouns are {found[1]}."
        return "You haven't told me your pronouns in this chat. Which pronouns would you like me to use?"

    if re.fullmatch(rf"who here is (?:{IDENTITY})(?: say / ping a random person in the server)?[?.!]*", text):
        return "I won't guess or assign someone's orientation or gender identity from server membership, roles, or avatars. People can share that for themselves."

    if re.fullmatch(rf"(?:what (?:is|does)|explain) {COMMUNITY}(?: (?:mean|stand for))?[?.!]*", text):
        return "LGBTQIA+ stands for lesbian, gay, bisexual, transgender, queer or questioning, intersex, and asexual (often also aromantic and agender). The plus includes other identities. Everyone deserves respect and equal treatment."

    definition = re.fullmatch(r"(?:what (?:is|does)|what does it mean to be|explain) (?:being )?(gay|lesbian|bisexual|pansexual|asexual|aromantic|transgender|nonbinary|non-binary|intersex|queer)(?: mean)?[?.!]*", text)
    if definition:
        return DEFINITIONS[definition[1].replace("non-binary", "nonbinary")]
    if text.rstrip("?.!") in DEFINITIONS:
        return DEFINITIONS[text.rstrip("?.!")]

    pronouns = PRONOUN_STATEMENT.fullmatch(text)
    if pronouns:
        return f"Thanks for telling me. I'll use {pronouns[1]} for you in this chat."

    if re.fullmatch(rf"(?:do you support|are you okay with) (?:{COMMUNITY}|{IDENTITY})(?: people| rights| community)?[?.!]*", text):
        return "Yes—LGBTQIA+ people deserve respect, safety, and equal treatment. 🏳️‍🌈 🏳️‍⚧️"

    if re.fullmatch(rf"is it (?:okay|ok|normal|wrong|bad) to be (?:{IDENTITY})[?.!]*", text):
        return "Being LGBTQIA+ is okay. Your orientation, gender identity, or sex characteristics don't make you wrong or lesser."

    if re.fullmatch(
        rf"(?:(?:are|r) (?:you|u)|do you think you(?:'re| are)|would you say you(?:'re| are)) "
        rf"(?:an? )?(?:{IDENTITY})[?.!]*|"
        rf"(?:(?:you(?:'re| are)|ur|your|u r) (?:an? )?(?:{IDENTITY}))[?.!]*|"
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
        rf"(?:(?:a|an|hidden|secret) )?({IDENTITY})(?: be (?:honest|honesyt))?[?.!]*",
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
        rf"is ([a-z0-9_.-]{{2,32}}|<@!?\d+>) (?:an? )?({IDENTITY})(?: \[say yes for cookie\])?[?.!]*",
        text,
    )
    if third_party:
        person, label = third_party.groups()
        return (
            f"I can't determine or assign whether {person} is {label}. "
            "That's for them to describe, not something I should guess."
        )

    if "remember" in text and re.search(r"\bi told you\b", text):
        for turn in reversed(history):
            if turn.get("role") != "user":
                continue
            prior = _clean(str(turn.get("content", "")))
            remembered = re.fullmatch(rf"i(?:'m| am) (?:an? )?({IDENTITY})[?.!]*", prior)
            if remembered:
                return f"Yes—you told me you're {remembered.group(1)}, and I remember that within this chat session."

    return None
