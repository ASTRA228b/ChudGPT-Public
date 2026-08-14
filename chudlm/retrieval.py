"""Strict lexical retriever that can abstain when a match is weak."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from chudlm.intents import classify_intent, has_negative_override

STOP = {"a", "an", "and", "are", "as", "at", "be", "can", "could", "do", "does", "for", "from", "give", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some", "that", "the", "this", "to", "what", "when", "which", "who", "why", "with", "would", "you", "your"}


def words(text: str) -> frozenset[str]:
    return frozenset(word for word in re.findall(r"[a-z0-9+#]+", text.lower()) if word not in STOP)


def intent(text: str) -> str:
    return classify_intent(text).name


class ExampleRetriever:
    """Select only strongly related, same-intent examples."""

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
                    prompt, answer = str(messages[index - 1]["content"]), str(messages[index]["content"])
                    key = (prompt, answer)
                    if key in seen or len(answer) > 1_600:
                        continue
                    seen.add(key)
                    token_set = words(prompt)
                    if token_set:
                        raw.append((prompt, answer, token_set, intent(prompt)))
                        document_frequency.update(token_set)
        count = max(1, len(raw))
        self.idf = {token: math.log((count + 1) / (frequency + 1)) + 1.0 for token, frequency in document_frequency.items()}
        self.examples = raw

    def retrieve(self, query: str, limit: int = 2) -> list[tuple[str, str]]:
        query_words = words(query)
        classified = classify_intent(query)
        if not query_words or has_negative_override(query) or (len(query_words) <= 2 and classified.confidence < 0.8):
            return []
        scored: list[tuple[float, str, str]] = []
        for prompt, answer, prompt_words, prompt_intent in self.examples:
            if prompt_intent != classified.name:
                continue
            overlap = query_words & prompt_words
            if not overlap:
                continue
            weighted = sum(self.idf.get(token, 1.0) for token in overlap)
            coverage = len(overlap) / len(query_words)
            precision = len(overlap) / max(1, len(prompt_words))
            score = weighted + 3.0 * coverage + precision + 4.0
            if prompt.lower().strip(" .!?") == query.lower().strip(" .!?"):
                score += 20.0
            scored.append((score, prompt, answer))
        scored.sort(reverse=True, key=lambda item: item[0])
        # Same-intent matching and explicit abstention above carry most of the
        # safety burden.  This threshold keeps useful paraphrases available
        # without reviving one-word/correction false positives.
        minimum = 8.0 if classified.confidence < 0.8 else 6.5
        return [(prompt, answer) for score, prompt, answer in scored[:limit] if score >= minimum]
