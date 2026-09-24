"""Conservative, whole-word racial-slur filtering for Public user input.

Keep identity terms and innocent substrings intact. This is a base-word
filter, not a classifier for hateful intent or every possible evasion.
"""

import re


_SLURS = re.compile(
    r"\b(?:niggers?|niggas?|kikes?|spics?|wetbacks?|beaners?|"
    r"ragheads?|towelheads?|zipperheads?|pakis?)\b",
    re.IGNORECASE,
)
SLUR_ONLY_REPLY = "Please send a question or message without racial slurs."


def filter_racial_slurs(text: str) -> tuple[str, bool]:
    cleaned, count = _SLURS.subn("", text)
    if not count:
        return text, False
    # Preserve line breaks and code indentation in the remaining request.
    cleaned = re.sub(r"(?<=\S)[ \t]{2,}", " ", cleaned).strip()
    return cleaned, True


def has_remaining_request(text: str) -> bool:
    return any(character.isalnum() for character in text)
