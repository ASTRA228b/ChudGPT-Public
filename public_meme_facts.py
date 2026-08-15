"""Small reviewed glossary for explicit meme questions and phrases.

This is intentionally not a general response database. Entries cover named
memes whose meaning is stable enough to explain briefly when the 21M neural
model produces an unrelated answer.
"""

from __future__ import annotations

import re

MEME_FACTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("tung tung tung sahur", "tung tung sahur"),
     "“Tung Tung Tung Sahur” is an absurdist 2025 Italian-brainrot meme built around a wooden character and a chant associated with waking people for sahur."),
    (("67 meme", "six seven meme", "6 7 meme"),
     "The “67” meme is intentionally context-light internet slang: people repeat the number as a joke, often because the lack of a clear meaning is the point."),
    (("skibidi toilet",),
     "Skibidi Toilet is a surreal YouTube animation meme series about singing toilet-headed characters fighting camera- and speaker-headed characters."),
    (("ohio meme", "only in ohio"),
     "The Ohio meme jokingly treats Ohio as a place where impossible or bizarre events are completely normal."),
    (("rizz",),
     "Rizz means charisma or skill at attracting someone; the word became a meme because people apply it to increasingly ridiculous situations."),
    (("gyatt",),
     "Gyatt is internet slang commonly used as an exaggerated reaction to someone's appearance. Its casual use can be crude, so context matters."),
    (("sigma meme", "sigma male"),
     "The sigma meme parodies the idea of an ultra-independent person who ignores social rules; online, it is usually ironic rather than serious advice."),
    (("rickroll", "rick roll"),
     "A rickroll is a bait-and-switch link that unexpectedly plays Rick Astley's “Never Gonna Give You Up.”"),
    (("among us", "sus meme"),
     "The Among Us meme uses “sus,” short for suspicious, and jokes about seeing the game's crewmate shape in ordinary objects."),
    (("this is fine",),
     "“This is fine” is the comic-panel meme of a dog calmly sitting in a burning room, used when someone pretends a clearly bad situation is manageable."),
    (("distracted boyfriend",),
     "Distracted Boyfriend is a stock-photo meme used to show someone abandoning one interest for a tempting new one."),
    (("woman yelling at cat",),
     "Woman Yelling at a Cat pairs an emotional reality-TV still with a confused cat at a dinner table to represent an absurd argument."),
    (("doge",),
     "Doge is the Shiba Inu meme captioned with deliberately broken phrases such as “such wow” and “very dog.”"),
    (("chud",),
     "“Chud” can be an insult for an unpleasant person and also references the film C.H.U.D. In this project, ChudGPT is simply the model family's name."),
)


def find_meme_fact(message: str) -> str | None:
    """Return a reviewed explanation only for an explicitly named meme."""
    normalized = " ".join(message.lower().replace("’", "'").split())
    for aliases, explanation in MEME_FACTS:
        # Whole-phrase boundaries prevent the `chud` glossary entry from
        # matching inside the product name `ChudGPT`.
        if any(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized) for alias in aliases):
            return explanation
    return None
