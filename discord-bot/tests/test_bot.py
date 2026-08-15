from bot import ChudGPTClient, clean_prompt, split_discord_message


def test_clean_prompt_removes_mentions_and_prefixes() -> None:
    assert clean_prompt("<@123> hello", 123, "!chud") == "hello"
    assert clean_prompt("<@!123> hello", 123, "!chud") == "hello"
    assert clean_prompt("!chud explain gravity", 123, "!chud") == "explain gravity"


def test_message_split_respects_discord_limit() -> None:
    chunks = split_discord_message("word " * 1_000, limit=200)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 200 for chunk in chunks)


def test_clear_uses_matching_public_api_endpoint(monkeypatch) -> None:
    client = ChudGPTClient("https://example.test/api/chat", 10)
    captured = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, bool]:
            return {"cleared": True}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(client.http, "post", fake_post)
    client.clear("discord-server-channel-user")
    assert captured == {
        "url": "https://example.test/api/clear",
        "json": {"session_id": "discord-server-channel-user"},
        "timeout": 10,
    }
