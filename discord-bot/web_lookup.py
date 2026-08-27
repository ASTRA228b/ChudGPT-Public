"""Bounded multi-source search and safe public-page reading for Discord."""

from __future__ import annotations

import re
import ipaddress
import socket
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import requests


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
DUCKDUCKGO_API = "https://api.duckduckgo.com/"
DUCKDUCKGO_HTML = "https://html.duckduckgo.com/html/"
STACKEXCHANGE_API = "https://api.stackexchange.com/2.3/search/advanced"
USER_AGENT = "ChudGPT-Public/20 Discord lookup (https://github.com/ASTRA228b/ChudGPT-Public)"


def parse_web_lookup(prompt: str) -> str | None:
    """Return the explicit lookup query, or None for ordinary conversation."""
    match = re.fullmatch(
        r"(?:web|wiki|wikipedia|search|lookup|look up)(?:\s+(?:for|about))?\s+(.{2,180})",
        re.sub(r"\s+", " ", prompt.strip()),
        re.I,
    )
    return match.group(1).strip() if match else None


def parse_public_url(prompt: str) -> str | None:
    """Extract one explicit public web link from a Discord message."""
    match = re.search(r"https?://[^\s<>]+", prompt, re.I)
    return match.group(0).rstrip(".,!?)]}\"'") if match else None


class _ReadableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title: list[str] = []
        self.text: list[str] = []
        self.metadata: dict[str, str] = {}
        self._hidden = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = {str(key).lower(): str(value) for key, value in attrs if value is not None}
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "form", "aside"}:
            self._hidden += 1
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            content = re.sub(r"\s+", " ", unescape(attributes.get("content", ""))).strip()
            if key in {"description", "og:description", "twitter:description", "og:title", "twitter:title"} and content:
                self.metadata.setdefault(key, content)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "form", "aside"} and self._hidden:
            self._hidden -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._hidden:
            return
        cleaned = re.sub(r"\s+", " ", data).strip()
        if cleaned:
            (self.title if self._in_title else self.text).append(cleaned)


class _SearchHTML(HTMLParser):
    """Extract a few ordinary DuckDuckGo results without executing scripts."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._field: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = {str(key).lower(): str(value) for key, value in attrs if value is not None}
        classes = set(attributes.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self.results.append({"title": "", "url": self._clean_url(attributes.get("href", "")), "snippet": ""})
            self._field, self._parts = "title", []
        elif self.results and ({"result__snippet", "result__snippet--highlight"} & classes):
            self._field, self._parts = "snippet", []

    def handle_endtag(self, tag: str) -> None:
        if self._field and tag in {"a", "div", "span"}:
            value = re.sub(r"\s+", " ", unescape(" ".join(self._parts))).strip()
            if self.results and value:
                self.results[-1][self._field] = value
            self._field, self._parts = None, []

    def handle_data(self, data: str) -> None:
        if self._field:
            self._parts.append(data)

    @staticmethod
    def _clean_url(value: str) -> str:
        absolute = urljoin(DUCKDUCKGO_HTML, unescape(value))
        parsed = urlparse(absolute)
        redirected = parse_qs(parsed.query).get("uddg", [])
        return unquote(redirected[0]) if redirected else absolute


class WikipediaLookup:
    def __init__(self, timeout: float = 7.0) -> None:
        self.timeout = timeout
        self.http = requests.Session()
        self.http.headers.update({"User-Agent": USER_AGENT})

    def lookup(self, query: str) -> str:
        original_query = query.strip()
        query = self._expand_query(original_query)
        sections: list[str] = []
        # DuckDuckGo's Instant Answer endpoint adds broader entities and
        # definitions without scraping search-result HTML.
        try:
            instant = self.http.get(
                DUCKDUCKGO_API,
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=self.timeout,
            )
            instant.raise_for_status()
            instant.encoding = "utf-8"
            instant_payload = instant.json()
            abstract = re.sub(r"\s+", " ", str(instant_payload.get("AbstractText", ""))).strip()
            abstract_url = str(instant_payload.get("AbstractURL", "")).strip()
            heading = str(instant_payload.get("Heading", query)).strip() or query
            if abstract and self._is_relevant(original_query, heading, abstract):
                sections.append(f"**{heading}** — {self._trim(abstract, 500)}" + (f"\n<{abstract_url}>" if abstract_url else ""))
        except (requests.RequestException, ValueError, KeyError):
            pass

        pages = []
        try:
            response = self.http.get(
                WIKIPEDIA_API,
                params={
                    "action": "query", "generator": "search", "gsrsearch": query,
                    "gsrlimit": 2, "prop": "extracts|info", "exintro": 1,
                    "explaintext": 1, "inprop": "url", "format": "json",
                    "formatversion": 2,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            response.encoding = "utf-8"
            payload = response.json()
            pages = payload.get("query", {}).get("pages", []) if isinstance(payload, dict) else []
        except (requests.RequestException, ValueError, KeyError):
            pass
        for page in pages[:2]:
            title = str(page.get("title", query))
            extract = re.sub(r"\s+", " ", str(page.get("extract", ""))).strip()
            url = str(page.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}")
            if (
                extract
                and self._is_relevant(original_query, title, extract)
                and url not in "\n".join(sections)
            ):
                sections.append(f"**Wikipedia: {title}** — {self._trim(extract, 420)}\n<{url}>")

        # The Instant Answer API intentionally covers only a subset of the
        # web. Explicit `web/search/lookup` commands may also use DDG's
        # script-free result page, giving the bot broader public-web reach
        # without keys, cookies, logins, or arbitrary background browsing.
        try:
            search = self.http.get(
                DUCKDUCKGO_HTML,
                params={"q": query},
                timeout=self.timeout,
                headers={"Accept": "text/html"},
            )
            search.raise_for_status()
            search.encoding = "utf-8"
            parser = _SearchHTML()
            parser.feed(getattr(search, "text", ""))
            existing = "\n".join(sections)
            for result in parser.results[:5]:
                title = result.get("title", "").strip()
                link = result.get("url", "").strip()
                snippet = result.get("snippet", "").strip()
                if not title or not link or link in existing:
                    continue
                if not urlparse(link).scheme in {"http", "https"}:
                    continue
                detail = f" — {self._trim(snippet, 260)}" if snippet else ""
                sections.append(f"**Web: {self._trim(title, 160)}**{detail}\n<{link}>")
                existing += "\n" + link
                if len(sections) >= 4:
                    break
        except (requests.RequestException, ValueError):
            pass

        # Stack Overflow is queried only for likely software questions, which
        # gives coding requests relevant live links without polluting normal
        # factual searches.
        if re.search(r"\b(?:code|python|c#|csharp|javascript|java|unity|error|exception|api|html|css|sql|program)\b", query, re.I):
            try:
                stack = self.http.get(
                    STACKEXCHANGE_API,
                    params={"site": "stackoverflow", "q": query, "pagesize": 2, "sort": "relevance"},
                    timeout=self.timeout,
                )
                stack.raise_for_status()
                stack.encoding = "utf-8"
                for item in stack.json().get("items", [])[:2]:
                    title = unescape(str(item.get("title", "Stack Overflow result")))
                    link = str(item.get("link", ""))
                    if link:
                        sections.append(f"**Stack Overflow:** {title}\n<{link}>")
            except (requests.RequestException, ValueError, KeyError):
                pass

        if not sections:
            return f"I couldn't find a useful live result for **{original_query}**. Try a more specific search."
        return "\n\n".join(sections[:4]) + "\n\n*Live web lookup; sources can change, so verify important claims.*"

    def read_url(self, url: str) -> str:
        """Read a small public text page while blocking common SSRF paths."""
        current = url
        for _redirect in range(4):
            self._validate_public_url(current)
            response = self.http.get(
                current, timeout=self.timeout, stream=True, allow_redirects=False,
                headers={"Accept": "text/html,text/plain,application/json,application/xml;q=0.8"},
            )
            if response.status_code in {301, 302, 303, 307, 308}:
                target = response.headers.get("Location")
                response.close()
                if not target:
                    raise ValueError("The page returned an empty redirect.")
                current = urljoin(current, target)
                continue
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            allowed = content_type.startswith("text/") or content_type in {"application/json", "application/xml", "application/xhtml+xml"}
            if not allowed:
                response.close()
                return f"I can open the link, but it is `{content_type or 'a binary file'}` rather than a readable text page."
            body = bytearray()
            truncated = False
            for chunk in response.iter_content(16_384):
                body.extend(chunk)
                if len(body) > 524_288:
                    del body[524_288:]
                    truncated = True
                    break
            response.close()
            charset = re.search(r"charset=([^;\s]+)", response.headers.get("Content-Type", ""), re.I)
            encoding = charset.group(1).strip('"\'') if charset else "utf-8"
            raw = bytes(body).decode(encoding, errors="replace")
            raw = self._repair_mojibake(raw)
            if "html" in content_type or "<html" in raw[:500].lower():
                parser = _ReadableHTML()
                parser.feed(raw)
                title = (
                    parser.metadata.get("og:title")
                    or parser.metadata.get("twitter:title")
                    or " ".join(parser.title).strip()
                    or urlparse(current).hostname
                    or "Web page"
                )
                description = (
                    parser.metadata.get("og:description")
                    or parser.metadata.get("twitter:description")
                    or parser.metadata.get("description")
                    or ""
                )
                combined_text = " ".join(parser.text)
                structured = re.search(
                    r"(?:content description|description)\s*:\s*(.{20,500}?)(?=\s+(?:file size|duration|dimensions|created|related|share url|embed details)\s*:|$)",
                    combined_text,
                    re.I,
                )
                if structured:
                    description = structured.group(1).strip()
                readable = self._summarize_page(description, parser.text)
            else:
                title = urlparse(current).hostname or "Web page"
                readable = self._summarize_page("", [raw])
            if not readable:
                return f"I opened **{title}**, but it did not contain readable page text."
            limit_note = " Read from the first 512 KB." if truncated else ""
            return f"**{self._trim(title, 180)}**\n**Summary:** {readable}\n<{current}>\n*Live page summary; verify important claims at the source.{limit_note}*"
        raise ValueError("The page redirected too many times.")

    @staticmethod
    def _validate_public_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only public HTTP or HTTPS links are supported.")
        if parsed.username or parsed.password or parsed.port not in {None, 80, 443}:
            raise ValueError("URLs with credentials or unusual ports are blocked.")
        hostname = parsed.hostname.rstrip(".").lower()
        if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
            raise ValueError("Local and private-network links are blocked.")
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))}
        except socket.gaierror as error:
            raise ValueError("The link's hostname could not be resolved.") from error
        if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
            raise ValueError("Local, private, reserved, and non-public addresses are blocked.")

    @staticmethod
    def _trim(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rsplit(" ", 1)[0] + "..."

    @staticmethod
    def _expand_query(query: str) -> str:
        """Disambiguate a few broad technical terms without changing intent."""
        expanded = re.sub(r"\bcsharp\b", "C# programming language", query, flags=re.I)
        if re.search(r"(?<!\w)c#(?!\w)", expanded, re.I):
            expanded = re.sub(r"(?<!\w)c#(?!\w)", "C# programming language", expanded, flags=re.I)
        if re.search(r"\brat\b", query, re.I) and re.search(
            r"\b(?:coding|computer|cyber|malware|security|hacking)\b", query, re.I
        ):
            expanded = "remote access trojan RAT cybersecurity definition"
        if re.search(r"\bskid\b", query, re.I) and re.search(
            r"\b(?:word|mean|coding|computer|community|hacking|slang)\b", query, re.I
        ):
            expanded = "script kiddie skid computer hacking slang definition"
        return expanded

    @staticmethod
    def _is_relevant(query: str, title: str, text: str) -> bool:
        """Reject obviously wrong entities returned for ambiguous searches."""
        haystack = f"{title} {text}".lower()
        lowered = query.lower()
        if re.search(r"\bcsharp\b|c#", lowered):
            return any(term in haystack for term in ("c#", "c sharp", "roslyn", ".net compiler"))
        if re.search(r"\brat\b", lowered) and re.search(r"\b(?:coding|computer|cyber|malware|security|hacking)\b", lowered):
            return "remote access trojan" in haystack or ("malware" in haystack and "rat" in haystack)
        if re.search(r"\bskid\b", lowered) and re.search(r"\b(?:word|mean|coding|computer|community|hacking|slang)\b", lowered):
            return "script kiddie" in haystack or ("hacker" in haystack and "skid" in haystack)
        words = {
            word for word in re.findall(r"[a-z0-9]{3,}", lowered)
            if word not in {"what", "does", "mean", "latest", "features", "about", "word", "search"}
        }
        return not words or any(word in haystack for word in words)

    @staticmethod
    def _repair_mojibake(text: str) -> str:
        if not any(marker in text for marker in ("â€", "Ã", "ðŸ")):
            return text
        try:
            return text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text

    @classmethod
    def _summarize_page(cls, description: str, chunks: list[str]) -> str:
        """Create a concise extractive summary without trusting the small LM."""
        candidates: list[str] = []
        low_value_description = re.fullmatch(
            r"(?:click|tap|open)\s+(?:here\s+)?to\s+(?:view|watch|read|open).{0,80}",
            description.strip(),
            re.I,
        )
        if description and not low_value_description:
            candidates.append(description)
        for chunk in chunks:
            cleaned = re.sub(r"\s+", " ", unescape(chunk)).strip()
            if len(cleaned) < 35:
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
                sentence = sentence.strip(" -|•")
                lowered = sentence.lower()
                if not 35 <= len(sentence) <= 500:
                    continue
                if any(phrase in lowered for phrase in (
                    "cookie", "privacy policy", "terms of service", "sign in", "sign up",
                    "copy link", "share to", "all rights reserved", "accept all",
                    "enable javascript", "subscribe to", "advertisement",
                    "translated based on your browser", "change the language",
                    "content description:", "file size:", "related gifs:",
                )):
                    continue
                candidates.append(sentence)

        unique: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = re.sub(r"[^a-z0-9]+", " ", candidate.lower()).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(candidate)
            if len(unique) == 3 or sum(map(len, unique)) >= 750:
                break
        if not unique:
            for chunk in chunks:
                cleaned = re.sub(r"\s+", " ", unescape(chunk)).strip()
                if 12 <= len(cleaned) <= 500:
                    unique.append(cleaned)
                    break
        return cls._trim(" ".join(unique), 900) if unique else ""
