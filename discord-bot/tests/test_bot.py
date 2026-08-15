from bot import ChudGPTClient, DISCORD_SYSTEM_PROMPT, add_recent_context, clean_prompt, discord_developer_reply, make_session_id, split_discord_message


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


def test_chat_enables_discord_context_without_changing_base_api(monkeypatch) -> None:
    client = ChudGPTClient("https://example.test/api/chat", 10)
    captured = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"reply": "Hello from Discord context."}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(client.http, "post", fake_post)
    assert client.chat("hello", "discord-session") == "Hello from Discord context."
    assert captured["json"]["context_mode"] == "discord"
    assert captured["json"]["system_instruction"] == DISCORD_SYSTEM_PROMPT
    assert captured["json"]["message"] == "hello"


def test_short_followup_uses_only_supplied_same_session_context() -> None:
    assert add_recent_context("the", ["click on the link"]) == (
        "Recent Discord context: click on the link\nCurrent message: the"
    )
    assert add_recent_context("tell me about the link", ["click on the link"]) == "tell me about the link"


def test_chat_falls_back_to_local_cuda_api(monkeypatch) -> None:
    client = ChudGPTClient("https://public.example/api/chat", 10)
    called = []

    class Response:
        def __init__(self, ok: bool) -> None:
            self.ok = ok

        def raise_for_status(self) -> None:
            if not self.ok:
                import requests
                raise requests.HTTPError("503")

        def json(self) -> dict[str, str]:
            return {"reply": "Local CUDA reply"}

    def fake_post(url, json, timeout):
        called.append(url)
        return Response(url.startswith("http://127.0.0.1:8010"))

    monkeypatch.setattr(client.http, "post", fake_post)
    assert client.chat("hello", "session") == "Local CUDA reply"
    assert called == ["https://public.example/api/chat", "http://127.0.0.1:8010/api/chat"]


def test_discord_developer_identity_is_stable() -> None:
    expected = "Astra (<@12345>) is ChudGPT's developer and the owner of this Discord bot."
    assert discord_developer_reply("Who is Astra?", 12345) == expected
    assert discord_developer_reply("Who made ChudGPT?", 12345) == expected
    assert discord_developer_reply("tell me about music", 12345) is None
