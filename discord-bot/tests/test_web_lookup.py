from __future__ import annotations

import pytest
import requests

from web_lookup import WikipediaLookup, parse_public_url, parse_web_lookup


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


def test_discord_message_extracts_public_link() -> None:
    assert parse_public_url("yo read https://example.com/page?q=1 please") == "https://example.com/page?q=1"
    assert parse_public_url("no link here") is None


@pytest.mark.parametrize("url", [
    "http://localhost/admin", "http://127.0.0.1/", "http://10.0.0.2/",
    "http://user:pass@example.com/", "https://example.com:8443/",
])
def test_url_reader_blocks_private_or_unusual_destinations(url: str) -> None:
    with pytest.raises(ValueError):
        WikipediaLookup._validate_public_url(url)


def test_url_reader_extracts_html_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status_code = 200
        headers = {"Content-Type": "text/html; charset=utf-8"}
        encoding = "utf-8"

        def raise_for_status(self) -> None: return None
        def iter_content(self, _size: int):
            yield b"<html><title>Test page</title><body><script>bad()</script><p>Hello useful world.</p></body></html>"
        def close(self) -> None: return None

    lookup = WikipediaLookup()
    monkeypatch.setattr("web_lookup.socket.getaddrinfo", lambda *_args: [(2, 1, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(lookup.http, "get", lambda *_args, **_kwargs: Response())
    reply = lookup.read_url("https://example.com/page")
    assert "Test page" in reply and "Hello useful world" in reply
    assert "bad()" not in reply


def test_url_reader_summarizes_metadata_and_drops_page_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status_code = 200
        headers = {"Content-Type": "text/html; charset=utf-8"}
        encoding = "utf-8"

        def raise_for_status(self) -> None: return None
        def iter_content(self, _size: int):
            yield (
                b'<html><head><title>Raw title</title>'
                b'<meta property="og:title" content="Gangster Thug GIF">'
                b'<meta property="og:description" content="An anime girl in a maid outfit makes a dramatic hand sign.">'
                b'</head><body><nav>Share to Facebook. Copy link. Privacy Policy.</nav>'
                b'<main><p>The short animation loops for three seconds.</p></main>'
                b'<footer>Terms of Service and all rights reserved.</footer></body></html>'
            )
        def close(self) -> None: return None

    lookup = WikipediaLookup()
    monkeypatch.setattr("web_lookup.socket.getaddrinfo", lambda *_args: [(2, 1, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(lookup.http, "get", lambda *_args, **_kwargs: Response())
    reply = lookup.read_url("https://example.com/gif")
    assert "**Gangster Thug GIF**" in reply
    assert "**Summary:** An anime girl" in reply
    assert "loops for three seconds" in reply
    assert "Share to Facebook" not in reply
    assert "Privacy Policy" not in reply
    assert "Terms of Service" not in reply


def test_url_reader_prefers_structured_content_description_over_click_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status_code = 200
        headers = {"Content-Type": "text/html"}
        encoding = "utf-8"

        def raise_for_status(self) -> None: return None
        def iter_content(self, _size: int):
            yield (
                b'<html><head><title>GIF page</title>'
                b'<meta property="og:description" content="Click to view the GIF"></head>'
                b'<body><p>Content Description: a purple-haired anime girl in a maid outfit makes a hand sign '
                b'File Size: 916KB Duration: 3 seconds Related GIFs: cats</p></body></html>'
            )
        def close(self) -> None: return None

    lookup = WikipediaLookup()
    monkeypatch.setattr("web_lookup.socket.getaddrinfo", lambda *_args: [(2, 1, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(lookup.http, "get", lambda *_args, **_kwargs: Response())
    reply = lookup.read_url("https://example.com/gif")
    assert "purple-haired anime girl" in reply
    assert "Click to view" not in reply
    assert "File Size" not in reply


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
    assert "Live web lookup" in reply


def test_web_lookup_degrades_cleanly_when_all_sources_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = WikipediaLookup()
    monkeypatch.setattr(lookup.http, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.Timeout()))
    assert "couldn't find a useful live result" in lookup.lookup("test")
