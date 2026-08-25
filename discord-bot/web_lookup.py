"""Small, explicit Wikipedia lookup tool for the Discord bot.

This deliberately uses one fixed HTTPS API.  It is not a general-purpose URL
fetcher and cannot be redirected to private/local hosts by Discord users.
"""

from __future__ import annotations

import re
from urllib.parse import quote

import requests


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "ChudGPT-Public/20 Discord lookup (https://github.com/ASTRA228b/ChudGPT-Public)"


def parse_web_lookup(prompt: str) -> str | None:
    """Return the explicit lookup query, or None for ordinary conversation."""
    match = re.fullmatch(
        r"(?:web|wiki|wikipedia|search|lookup|look up)(?:\s+(?:for|about))?\s+(.{2,180})",
        re.sub(r"\s+", " ", prompt.strip()),
        re.I,
    )
    return match.group(1).strip() if match else None


class WikipediaLookup:
    def __init__(self, timeout: float = 7.0) -> None:
        self.timeout = timeout
        self.http = requests.Session()
        self.http.headers.update({"User-Agent": USER_AGENT})

    def lookup(self, query: str) -> str:
        response = self.http.get(
            WIKIPEDIA_API,
            params={
                "action": "query", "generator": "search", "gsrsearch": query,
                "gsrlimit": 1, "prop": "extracts|info", "exintro": 1,
                "explaintext": 1, "inprop": "url", "format": "json",
                "formatversion": 2,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        response.encoding = "utf-8"
        payload = response.json()
        pages = payload.get("query", {}).get("pages", []) if isinstance(payload, dict) else []
        if not pages:
            return f"I couldn't find a Wikipedia result for **{query}**."
        page = pages[0]
        title = str(page.get("title", query))
        extract = re.sub(r"\s+", " ", str(page.get("extract", ""))).strip()
        if len(extract) > 900:
            extract = extract[:897].rsplit(" ", 1)[0] + "..."
        url = str(page.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}")
        summary = extract or "Wikipedia returned the page but no short introduction."
        return f"**{title}** — {summary}\n<{url}>\n*Live Wikipedia lookup; verify important or current claims.*"
