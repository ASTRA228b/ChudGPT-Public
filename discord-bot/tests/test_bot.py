from bot import clean_prompt, split_discord_message


def test_clean_prompt_removes_mentions_and_prefixes() -> None:
    assert clean_prompt("<@123> hello", 123, "!chud") == "hello"
    assert clean_prompt("<@!123> hello", 123, "!chud") == "hello"
    assert clean_prompt("!chud explain gravity", 123, "!chud") == "explain gravity"


def test_message_split_respects_discord_limit() -> None:
    chunks = split_discord_message("word " * 1_000, limit=200)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 200 for chunk in chunks)
