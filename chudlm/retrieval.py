"""Small lexical retriever for Public's project-authored few-shot examples."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

STOP = {
    "a", "an", "and", "are", "as", "at", "be", "can", "could", "do", "does", "for",
    "from", "give", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "not", "please", "some", "that", "the", "this", "to", "what", "when", "which", "who",
    "why", "with", "would", "you", "your",
}


def words(text: str) -> frozenset[str]:
    return frozenset(word for word in re.findall(r"[a-z0-9+#]+", text.lower()) if word not in STOP)


def intent(text: str) -> str:
    lowered = text.lower()
    tokens = words(text)
    if any(token in tokens for token in ("code", "python", "c#", "csharp", "javascript", "unity", "sql", "debug", "program", "script", "rust", "go")):
        return "code"
    if any(token in tokens for token in ("meme", "rickroll", "brainrot", "amogus", "aura", "67")):
        return "meme"
    if re.search(r"\d\s*(?:[+*/×÷]|-(?=\s*\d))\s*\d", lowered) or any(token in tokens for token in ("calculate", "mph", "percent")):
        return "math"
    if any(phrase in lowered for phrase in ("who are you", "what are you", "your name", "chudgpt", "your abilities", "what can you do")):
        return "identity"
    if re.match(r"^(?:hi|hello|hey|yo)\b", lowered):
        return "conversation"
    if lowered.rstrip().endswith("?"):
        return "question"
    return "conversation"


class ExampleRetriever:
    """Select related examples using TF-IDF-like lexical overlap and intent."""

    def __init__(self, paths: tuple[Path, ...]) -> None:
        raw: list[tuple[str, str, frozenset[str], str]] = []
        seen: set[tuple[str, str]] = set()
        document_frequency: Counter[str] = Counter()
        for path in paths:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                messages = json.loads(line)["messages"]
                for index in range(1, len(messages)):
                    if messages[index - 1]["role"] != "user" or messages[index]["role"] != "assistant":
                        continue
                    prompt = str(messages[index - 1]["content"])
                    answer = str(messages[index]["content"])
                    key = (prompt, answer)
                    if key in seen or len(answer) > 1_600:
                        continue
                    seen.add(key)
                    token_set = words(prompt)
                    if not token_set:
                        continue
                    raw.append((prompt, answer, token_set, intent(prompt)))
                    document_frequency.update(token_set)
        count = max(1, len(raw))
        self.idf = {token: math.log((count + 1) / (frequency + 1)) + 1.0 for token, frequency in document_frequency.items()}
        self.examples = raw

    def retrieve(self, query: str, limit: int = 2) -> list[tuple[str, str]]:
        query_words = words(query)
        if not query_words:
            return []
        query_intent = intent(query)
        scored: list[tuple[float, str, str]] = []
        for prompt, answer, prompt_words, prompt_intent in self.examples:
            overlap = query_words & prompt_words
            if not overlap:
                continue
            weighted = sum(self.idf.get(token, 1.0) for token in overlap)
            coverage = len(overlap) / max(1, len(query_words))
            precision = len(overlap) / max(1, len(prompt_words))
            score = weighted + 3.0 * coverage + precision
            score += 4.0 if prompt_intent == query_intent else -3.0
            if prompt.lower().strip(" .!?") == query.lower().strip(" .!?"):
                score += 20.0
            scored.append((score, prompt, answer))
        scored.sort(reverse=True, key=lambda item: item[0])
        return [(prompt, answer) for score, prompt, answer in scored[:limit] if score > 2.5]
