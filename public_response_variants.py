"""Wording variation for existing grounded routes, never generated-answer repair."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from chudlm.emoji_awareness import strip_emoji_context


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", strip_emoji_context(text).casefold().replace("’", "'")).strip().rstrip("?.!")


def vary_grounded_response(message: str, answer: str, reason: str | None,
                           history: Sequence[Mapping[str, str]]) -> str:
    """Rotate only on repeat requests in this session. First answers stay stable.

    No lookup of generated text, model-name detection, or generic nonanswer.
    Exact math and unrecognized grounded formats are returned unchanged.
    """
    if reason not in {"geography", "emoji_semantics", "lgbtq_identity", "project_identity", "canned_greeting"}:
        return answer
    repeats = sum(turn.get("role") == "user" and _key(str(turn.get("content", ""))) == _key(message) for turn in history)
    if not repeats:
        return answer
    alternatives = [answer]
    if reason == "geography":
        capital = re.fullmatch(r"The capital of (.+?) is (.+)\.", answer)
        if capital:
            place, city = capital.groups()
            alternatives.extend([f"{city} is the capital of {place}.", f"For {place}, the capital is {city.rstrip('.')}."])
    elif reason == "canned_greeting" and _key(message) in {"hi", "hello", "hey", "yo", "hi there", "hey there"}:
        alternatives.extend(["Hey! What would you like to talk about?", "Hi again. What's on your mind?", "Hello! What are we getting into today?"])
    elif reason == "emoji_semantics" and answer.startswith("That reads as "):
        meaning = answer.removeprefix("That reads as ").removesuffix(". What's the context?")
        alternatives.extend([f"Possible meanings include {meaning}. The surrounding message matters.", f"I'd read that as {meaning}, depending on the context."])
    elif reason == "lgbtq_identity":
        if answer == "Thanks for telling me. I'll respect how you describe yourself.":
            alternatives.extend(["Got it. Thanks for sharing that with me.", "I hear you. I'll respect the identity you use for yourself."])
        elif "LGBTQIA+ people deserve respect, safety, and equal treatment" in answer:
            alternatives.extend(["Yes. LGBTQIA+ people deserve dignity and equal treatment.", "I support treating LGBTQIA+ people with respect and care."])
        elif answer.startswith("No—I'm an AI"):
            alternatives.extend(["I'm an AI, so I don't experience attraction or have a gender identity. I can discuss LGBTQIA+ topics, though.", "I don't have a personal orientation—I'm a language model. That doesn't stop us talking about the topic."])
    elif reason == "project_identity":
        if answer.startswith("I'm ChudGPT-Public V20,"):
            alternatives.extend([answer.replace("I'm ChudGPT-Public V20,", "You're chatting with ChudGPT-Public V20,", 1), answer.replace("I'm ChudGPT-Public V20,", "My name is ChudGPT-Public V20; I'm", 1)])
        elif answer.startswith("ChudGPT is Astra's overall project and model family."):
            alternatives.append(answer.replace("ChudGPT is Astra's overall project and model family.", "Astra's ChudGPT project includes a family of experimental models.", 1))
    return alternatives[repeats % len(alternatives)]
