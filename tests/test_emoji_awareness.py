"""Broad emoji-awareness and false-positive regression coverage."""

from __future__ import annotations

import pytest

from chudlm.emoji_awareness import (
    add_emoji_context,
    emoji_annotations,
    emoji_database,
    strip_emoji_context,
)
from chudlm.text_normalization import normalize_user_text


def test_database_covers_current_full_unicode_metadata() -> None:
    database = emoji_database()
    assert database.sequence_count >= 5_200
    assert float(database.max_emoji_version or 0) >= 17.0


@pytest.mark.parametrize(
    ("sequence", "name_fragment"),
    [
        ("😀", "face"), ("👍🏽", "thumb"), ("❤️", "heart"),
        ("🐶", "dog"), ("🍕", "pizza"), ("✈️", "airplane"),
        ("⚽", "ball"), ("💻", "computer"), ("✅", "check"),
        ("🇧🇷", "brazil"),
    ],
)
def test_major_emoji_categories_are_recognized(sequence: str, name_fragment: str) -> None:
    record = emoji_database().get(sequence)
    assert record is not None
    searchable = " ".join((record.name, *record.aliases)).lower()
    assert name_fragment in searchable


@pytest.mark.parametrize(
    "sequence",
    ["👨‍💻", "👩‍💻", "🧑‍💻", "👨‍🚀", "👩‍🚀", "👨‍👩‍👧‍👦", "❤️‍🔥", "❤️‍🩹"],
)
def test_zwj_sequences_are_recognized_as_whole_emoji(sequence: str) -> None:
    record = emoji_database().get(sequence)
    assert record is not None and record.is_zwj_sequence
    assert len(emoji_annotations(sequence)) == 1


@pytest.mark.parametrize("alias", ["sob", "skull", "fire", "heart", "red_heart"])
def test_common_colon_aliases_resolve(alias: str) -> None:
    record = emoji_database().from_alias(alias)
    assert record is not None
    assert emoji_annotations(f":{alias}:")


def test_discord_custom_emoji_uses_name_but_never_exposes_id() -> None:
    annotations = emoji_annotations("<:chud_laugh:123456789> <a:spin:987654321>")
    joined = " ".join(annotations)
    assert "custom emoji=chud laugh" in joined
    assert "animated custom emoji=spin" in joined
    assert "123456789" not in joined and "987654321" not in joined


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("bro what 😭💀", "humorous"),
        ("my dog died 😭", "sadness"),
        ("the update is amazing 🔥", "praise"),
        ("Thanks 🙏", "thanks"),
        ("🤨", "skepticism"),
    ],
)
def test_contextual_meaning_beats_one_fixed_definition(message: str, expected: str) -> None:
    assert expected in " ".join(emoji_annotations(message)).lower()


@pytest.mark.parametrize("message", [":D", ":)", ":(", ";)", ":P", "XD", ":/", ":|", "<3", "¯\\_(ツ)_/¯"])
def test_classic_emoticons_are_recognized(message: str) -> None:
    assert emoji_annotations(message)


@pytest.mark.parametrize(
    "message",
    [
        "https://example.com/a:b", "C:\\Users\\test\\file.py", "```json\n{\"x\": \":D\"}\n```",
        "<@123456789>", "<@&123456789>", "<#123456789>", "12:30", "a:b", "namespace::Type",
    ],
)
def test_code_urls_mentions_and_punctuation_do_not_false_trigger(message: str) -> None:
    assert emoji_annotations(message) == []


def test_model_annotation_preserves_original_and_is_reversible() -> None:
    original = "bro 😭💀"
    annotated = add_emoji_context(original)
    assert annotated.startswith(original)
    assert "[emoji context:" in annotated
    assert strip_emoji_context(annotated) == original


def test_normalization_expands_slang_without_dropping_emoji() -> None:
    normalized = normalize_user_text("hru 😭")
    assert normalized.startswith("how are you 😭")
    assert "emoji context" in normalized
    assert normalize_user_text("hru 😭", include_emoji_hints=False) == "how are you 😭"


def test_annotation_is_compact_and_deduplicated() -> None:
    annotations = emoji_annotations("😭😭😭 💀💀 🔥🔥 👍🏽 ❤️ 🇺🇸 🐶", limit=4)
    assert len(annotations) <= 4
    assert len(annotations) == len(set(annotations))
