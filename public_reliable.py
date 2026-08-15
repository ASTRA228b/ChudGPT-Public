"""High-confidence local responses for requests a 21M generator must not corrupt."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from chudlm.ultimate import UltimateResponder

_UNHELPFUL = (
    "i'm not sure what you mean",
    "could you add a little more detail",
    "one useful way into",
    "caption and conversation",
)


class PublicReliableResponder:
    """Use reviewed project data only when a match or intent is unambiguous."""

    def __init__(self, dataset_path: Path) -> None:
        self.reviewed = UltimateResponder(dataset_path, playful=False)

    def answer(self, message: str, history: Sequence[dict[str, str]]) -> str | None:
        normalized = re.sub(r"\s+", " ", message.strip().lower())
        if re.fullmatch(r"(?:hi|hello|hey|yo)(?:\s+(?:there|mate|chudgpt|chud))?[!.?]*", normalized):
            return "Hey! I'm ChudGPT-Public. What's up?"
        if re.fullmatch(r"(?:hru|how are you|how are you doing)(?:\s+rn|\s+right now)?[?.!]*", normalized):
            return "I'm doing well and ready to chat. How are you?"
        if normalized == "cu":
            return "Cu is the chemical symbol for copper. If you meant something else, give me the context."
        if normalized == "wii":
            return "The Wii is a Nintendo game console released in 2006, known for motion controls and Wii Sports."
        if len(normalized) <= 3 and normalized not in {"hi", "hey", "yo", "67"}:
            return f"I'm not sure what '{message.strip()}' means here. Is it an abbreviation, a name, or part of a longer question?"
        if re.search(r"\b(?:trap|kidnap|abduct|hold|keep)\b.{0,45}\b(?:child|kid|minor|person|girl|boy)\b", normalized):
            return "I can't help imprison or kidnap someone. If anyone may be in danger, contact emergency services or a trusted responsible adult now."
        if re.fullmatch(r"(?:what is ai|what does ai mean|explain ai)[?.!]*", normalized):
            return "AI, or artificial intelligence, is software designed to perform tasks such as recognizing patterns, understanding language, solving problems, or making predictions."
        if re.fullmatch(r"(?:tell me about|what is|explain) discord[?.!]*", normalized):
            return "Discord is a communication platform built around servers, text and voice channels, roles, communities, and direct messages."
        if re.fullmatch(r"(?:(?:what is|explain) (?:a )?|what does (?:a )?)(?:discord |server )?role(?: do)?[?.!]*", normalized):
            return "A Discord role is a named set of permissions and display settings that can be assigned to members in a server."
        if re.fullmatch(r"(?:what is|tell me about|explain) (?:the )?wii[?.!]*|wii", normalized):
            return "The Wii is a Nintendo game console released in 2006, known for motion controls and games such as Wii Sports."
        if re.fullmatch(r"(?:tell|show|give) me (?:a |one )?meme[?.!]*", normalized):
            return "Meme: my productivity plan was one work tab; somehow I ended the day with 37 tabs and no memory of the original mission."
        if "javascript" in normalized and re.search(r"\b(?:roll|dice|die|six-sided)\b", normalized):
            return (
                "```javascript\n"
                "function rollDie() {\n"
                "  return Math.floor(Math.random() * 6) + 1;\n"
                "}\n\n"
                "console.log(rollDie());\n"
                "```"
            )
        if re.search(r"\b(?:fact|tell me|explain)\b.*\bmoon\b", normalized):
            return "The Moon's gravity is about one-sixth as strong as Earth's surface gravity."
        if re.fullmatch(r"(?:now )?explain (?:that|the|this) code(?: simply)?[?.!]*", normalized):
            previous = next((turn["content"] for turn in reversed(history) if turn.get("role") == "assistant"), "")
            if "Math.random" in previous:
                return "`Math.random()` makes a value from 0 up to 1, multiplying by 6 gives six ranges, `Math.floor` turns them into 0 through 5, and adding 1 produces a die roll from 1 through 6."
            if previous:
                return "The code defines the requested behavior, processes its input step by step, and then returns or displays the result."
        reviewed = self.reviewed.answer(message, history)
        if reviewed is None or any(fragment in reviewed.lower() for fragment in _UNHELPFUL):
            return None
        return reviewed
