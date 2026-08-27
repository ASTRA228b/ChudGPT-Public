from __future__ import annotations

import json
from pathlib import Path

from music_instructions import MUSIC_MODEL_NAME, MUSIC_SYSTEM_PROMPT
from public_api_server import MusicReliableResponder


def test_music_identity_and_funny_personality() -> None:
    responder = MusicReliableResponder()
    reply = responder.answer("What are you?", [])
    assert reply is not None
    assert MUSIC_MODEL_NAME in reply
    assert "music" in reply.lower()
    assert "funny" in MUSIC_SYSTEM_PROMPT.lower()


def test_music_copyright_boundary_is_helpful() -> None:
    reply = MusicReliableResponder().answer("Give me the full lyrics to Bohemian Rhapsody", [])
    assert reply is not None
    assert "can't provide" in reply.lower()
    assert "original" in reply.lower()


def test_music_dataset_is_large_and_unique() -> None:
    path = Path(__file__).parents[1] / "data" / "music_v1_conversations.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    serialized = {json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records}
    assert len(records) >= 700
    assert len(serialized) >= 700
    assert any("funny" in json.dumps(record).lower() or "ridiculous" in json.dumps(record).lower() for record in records)
