"""Conservative model-facing normalization for casual chat language."""

from __future__ import annotations

import re

EXPANSIONS = {
    "hru": "how are you",
    "wbu": "what about you",
    "wyd": "what are you doing",
    "idk": "I do not know",
    "imo": "in my opinion",
    "imho": "in my honest opinion",
    "ngl": "not going to lie",
    "rn": "right now",
    "btw": "by the way",
    "bc": "because",
    "cuz": "because",
    "pls": "please",
    "plz": "please",
    "u": "you",
    "ur": "your",
    "r": "are",
    "yk": "you know",
    "lmk": "let me know",
    "nvm": "never mind",
    "fr": "for real",
    "frfr": "for real",
    "tbh": "to be honest",
    "afaik": "as far as I know",
    "irl": "in real life",
    "fyi": "for your information",
    "jk": "just kidding",
    "dw": "do not worry",
    "ty": "thank you",
    "thx": "thanks",
    "np": "no problem",
    "omw": "on my way",
    "afk": "away from keyboard",
    "brb": "be right back",
    "gtg": "got to go",
    "idc": "I do not care",
    "ik": "I know",
    "ikr": "I know, right",
    "mb": "my bad",
    "smt": "something",
}

CORRECTIONS = {
    "dose": "does",
    "dosen't": "doesn't",
    "doesnt": "doesn't",
    "enything": "anything",
    "somthing": "something",
    "actullay": "actually",
    "relly": "really",
    "becuase": "because",
    "langauge": "language",
    "infomration": "information",
    "convo": "conversation",
    "tho": "though",
    "ppl": "people",
    "msg": "message",
}


def normalize_user_text(text: str) -> str:
    """Expand only well-known whole-token shorthand; never rewrite code."""
    if "```" in text or re.search(r"https?://|\b(?:class|def|function|using)\s+\w+", text):
        return text.strip()

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        lowered = token.casefold()
        replacement = EXPANSIONS.get(lowered, CORRECTIONS.get(lowered, token))
        if token[:1].isupper():
            replacement = replacement[:1].upper() + replacement[1:]
        return replacement

    normalized = re.sub(r"\b[A-Za-z']+\b", replace, text.strip())
    return re.sub(r"\s+", " ", normalized)
