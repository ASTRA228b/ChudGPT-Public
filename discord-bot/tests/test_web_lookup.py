from __future__ import annotations

import pytest
import requests

from web_lookup import WikipediaLookup, parse_web_lookup


@pytest.mark.parametrize("prompt, expected", [
    ("web gorilla tag", "gorilla tag"),
    ("search for Saturn", "Saturn"),
    ("look up Ada Lovelace", "Ada Lovelace"),
    ("wikipedia Python programming", "Python programming"),
    ("hello", None),
    ("open https://example.com", None),
])
def test_parse_web_lookup_is_explicit(prompt: str, expected: str | None) -> None:
    assert parse_web_lookup(prompt) == expected


def test_wikipedia_lookup_uses_fixed_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"query": {"pages": [{
                "title": "Gorilla Tag", "extract": "Gorilla Tag is a virtual reality game.",
                "fullurl": "https://en.wikipedia.org/wiki/Gorilla_Tag",
            }]}}

    seen: dict = {}

    def fake_get(url: str, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return Response()

    lookup = WikipediaLookup()
    monkeypatch.setattr(lookup.http, "get", fake_get)
    reply = lookup.lookup("gorilla tag")
    assert seen["url"] == "https://en.wikipedia.org/w/api.php"
    assert seen["params"]["gsrsearch"] == "gorilla tag"
    assert "Gorilla Tag is a virtual reality game" in reply
    assert "Live Wikipedia lookup" in reply


def test_wikipedia_lookup_does_not_hide_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = WikipediaLookup()
    monkeypatch.setattr(lookup.http, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.Timeout()))
    with pytest.raises(requests.Timeout):
        lookup.lookup("test")
