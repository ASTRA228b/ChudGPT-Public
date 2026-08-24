"""Compact Unicode and Discord emoji semantics for the small Public checkpoint."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import emoji as emoji_lib

DISCORD_CUSTOM_RE = re.compile(r"<(?P<animated>a?):(?P<name>[A-Za-z0-9_]{2,32}):(?P<id>\d{2,22})>")
COLON_ALIAS_RE = re.compile(r"(?<![\w/]):(?P<name>[A-Za-z0-9_+\-]{2,40}):(?![/\w])")
SKIN_TONES = tuple(chr(value) for value in range(0x1F3FB, 0x1F400))

INTERNET_HINTS = {
    "😭": "sadness, intense laughter, disbelief, or dramatic frustration",
    "💀": "death/skull; online often extremely funny, disbelief, or embarrassment",
    "🔥": "literal fire, heat, excitement, praise, or impressive quality",
    "🙏": "thanks, asking, hoping, praying, or pleading",
    "💯": "one hundred, strong agreement, authenticity, or excellence",
    "🤨": "skepticism, questioning, or suspicion",
    "😐": "neutral, unimpressed, awkward, or mildly annoyed",
    "🤡": "clown, foolish behavior, or self-mockery",
    "🗿": "moai; online often a deadpan or absurd meme reaction",
    "👀": "looking, attention, curiosity, or anticipation",
    "❤️": "love, affection, care, or strong approval",
}

COMBINATION_HINTS = {
    "💀😭": "strong humorous disbelief or laughter",
    "😭💀": "strong humorous disbelief or laughter",
    "😭🙏": "intense emotion with pleading, thanks, hope, or relief",
    "💀🙏": "humorous disbelief mixed with pleading or thanks",
    "😭✌️": "dramatic emotion with a playful sign-off",
    "🗿🍷": "deadpan, mock-refined meme reaction",
    "🤨📸": "suspicion or catching something questionable",
    "👀🔥": "attention or anticipation toward something impressive",
}

EMOTICONS = {
    ":D": "big smile or laughter", ":)": "smile", ":(": "sadness",
    ";)": "wink", ":P": "playful teasing", "XD": "laughter",
    "xD": "laughter", ":/": "uncertainty or awkwardness", ":|": "neutral reaction",
    "<3": "heart or affection", "¯\\_(ツ)_/¯": "shrug or uncertainty",
}


@dataclass(frozen=True)
class EmojiMetadata:
    sequence: str
    name: str
    category: str
    aliases: tuple[str, ...]
    usage_hint: str | None
    supports_modifiers: bool
    is_zwj_sequence: bool
    emoji_version: float | int | None


def _plain_name(value: str) -> str:
    return value.strip(":").replace("_", " ").replace("-", " ")


def _category(sequence: str, name: str) -> str:
    """Assign a compact major Unicode-style category from CLDR-compatible names."""
    words = set(name.split())
    if "flag" in words or all(0x1F1E6 <= ord(char) <= 0x1F1FF for char in sequence if char != "\ufe0f"):
        return "Flags"
    if any(term in name for term in ("heart", "face", "emotion", "smile", "cry", "angry", "skull", "kiss")):
        return "Smileys & Emotion"
    if any(term in name for term in ("hand", "finger", "person", "people", "woman", "man", "child", "baby", "body", "ear", "eye", "leg", "arm", "thumb")):
        return "People & Body"
    if any(term in name for term in ("cat", "dog", "fox", "frog", "bird", "penguin", "animal", "tree", "flower", "moon", "sun", "weather", "plant", "insect")):
        return "Animals & Nature"
    if any(term in name for term in ("food", "drink", "apple", "pizza", "burger", "cookie", "bowl", "rice", "bread", "cake", "fruit", "vegetable", "cup")):
        return "Food & Drink"
    if any(term in name for term in ("car", "airplane", "rocket", "house", "building", "train", "bus", "ship", "map", "statue", "travel")):
        return "Travel & Places"
    if any(term in name for term in ("ball", "game", "sport", "guitar", "trophy", "medal", "activity", "skate", "ski")):
        return "Activities"
    if any(term in name for term in ("check", "cross", "warning", "question", "exclamation", "arrow", "button", "symbol", "zodiac", "number")):
        return "Symbols"
    return "Objects"


class EmojiDatabase:
    """Cached view over emoji 2.15's full Unicode Emoji 17.0 sequence data."""

    def __init__(self) -> None:
        self._records: dict[str, EmojiMetadata] = {}
        self._aliases: dict[str, str] = {}
        for sequence, raw in emoji_lib.EMOJI_DATA.items():
            canonical = str(raw.get("en", ":emoji:"))
            aliases = tuple(str(alias).strip(":") for alias in raw.get("alias", ()))
            name = _plain_name(canonical)
            base = "".join(char for char in sequence if char not in SKIN_TONES and char != "\ufe0f")
            record = EmojiMetadata(
                sequence=sequence,
                name=name,
                category=_category(sequence, name),
                aliases=aliases,
                usage_hint=INTERNET_HINTS.get(sequence) or INTERNET_HINTS.get(base),
                supports_modifiers=any(tone in sequence for tone in SKIN_TONES) or any(
                    sequence + tone in emoji_lib.EMOJI_DATA for tone in SKIN_TONES
                ),
                is_zwj_sequence="\u200d" in sequence,
                emoji_version=raw.get("E"),
            )
            self._records[sequence] = record
            self._aliases[canonical.strip(":").casefold()] = sequence
            for alias in aliases:
                self._aliases[alias.casefold()] = sequence

    @property
    def sequence_count(self) -> int:
        return len(self._records)

    @property
    def max_emoji_version(self) -> float | int | None:
        versions = [record.emoji_version for record in self._records.values() if record.emoji_version is not None]
        return max(versions, default=None)

    def get(self, sequence: str) -> EmojiMetadata | None:
        return self._records.get(sequence)

    def from_alias(self, alias: str) -> EmojiMetadata | None:
        sequence = self._aliases.get(alias.strip(":").casefold())
        return self.get(sequence) if sequence else None


@lru_cache(maxsize=1)
def emoji_database() -> EmojiDatabase:
    return EmojiDatabase()


def _context_hint(text: str, sequences: list[str]) -> str | None:
    cluster = "".join(sequences)
    compact = cluster.replace("\ufe0f", "")
    for known, hint in COMBINATION_HINTS.items():
        if known.replace("\ufe0f", "") in compact:
            return hint
    lowered = text.casefold()
    if any(term in lowered for term in ("died", "dead dog", "funeral", "grief", "miss them", "hurt", "failed the test")):
        return "likely sadness, grief, or frustration in this context"
    if any(term in lowered for term in ("lol", "lmao", "bro", "joke", "funny", "what is this")) and any(
        item in sequences for item in ("😭", "💀", "🤣", "😂")
    ):
        return "likely a humorous or disbelieving reaction in this context"
    if any(term in lowered for term in ("fixed", "amazing", "great", "cooked", "update", "win")) and "🔥" in sequences:
        return "likely praise, excitement, or success in this context"
    return None


def emoji_annotations(text: str, *, include_discord: bool = True, limit: int = 4) -> list[str]:
    """Return compact semantics for only the emoji constructs in this message."""
    db = emoji_database()
    annotations: list[str] = []
    sequences: list[str] = []
    seen: set[str] = set()
    for match in emoji_lib.emoji_list(text):
        sequence = match["emoji"]
        if sequence in seen:
            continue
        seen.add(sequence)
        sequences.append(sequence)
        record = db.get(sequence)
        if record:
            detail = record.name
            if record.usage_hint:
                detail += f"; {record.usage_hint}"
            annotations.append(f"{sequence}={detail}")
    if include_discord:
        for match in DISCORD_CUSTOM_RE.finditer(text):
            name = _plain_name(match.group("name"))
            marker = "animated custom emoji" if match.group("animated") else "custom emoji"
            annotations.append(f"{marker}={name}")
        for match in COLON_ALIAS_RE.finditer(text):
            record = db.from_alias(match.group("name"))
            if record:
                annotations.append(f":{match.group('name')}:={record.name}")
    if not re.search(r"https?://|```|[A-Za-z]:\\", text):
        for symbol, meaning in EMOTICONS.items():
            if re.search(rf"(?<!\S){re.escape(symbol)}(?!\S)", text):
                annotations.append(f"emoticon {symbol}={meaning}")
    context = _context_hint(text, sequences)
    if context:
        # The combined interpretation already captures tone. Keep individual
        # entries to their names so a two-emoji reaction does not triple the
        # prompt cost with overlapping explanations.
        annotations = [entry.split(";", 1)[0] for entry in annotations]
        annotations.insert(0, f"combined tone={context}")
    # Deduplicate while keeping the most useful message-order hints.
    return list(dict.fromkeys(annotations))[:limit]


def add_emoji_context(text: str, *, include_discord: bool = True) -> str:
    """Preserve original text and append a small private semantic annotation."""
    if "```" in text:
        return text
    hints = emoji_annotations(text, include_discord=include_discord)
    if not hints:
        return text
    return f"{text} [emoji context: {' | '.join(hints)}]"


def strip_emoji_context(text: str) -> str:
    """Remove only the private suffix produced by :func:`add_emoji_context`."""
    return re.sub(r"\s+\[emoji context:.*\]\s*$", "", text, flags=re.S).strip()
