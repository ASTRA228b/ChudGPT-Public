from __future__ import annotations

import json
from pathlib import Path

from music_instructions import MUSIC_MODEL_NAME, MUSIC_SYSTEM_PROMPT
from public_api_server import MusicModelService


def test_music_identity_and_funny_personality_is_learned_not_hardcoded() -> None:
    assert MUSIC_MODEL_NAME in MUSIC_SYSTEM_PROMPT
    assert "original songwriting" in MUSIC_SYSTEM_PROMPT.lower()
    assert "funny" in MUSIC_SYSTEM_PROMPT.lower()
    source = (Path(__file__).parents[1] / "public_api_server.py").read_text(encoding="utf-8")
    assert "class MusicReliableResponder" not in source
    assert "self.reliable = None" in source


def test_music_chat_bypasses_every_public_answer_router() -> None:
    source = (Path(__file__).parents[1] / "public_api_server.py").read_text(encoding="utf-8")
    music_source = source[source.index("class MusicModelService"):source.index("def create_app(")]
    assert "exact_instruction_response(" not in music_source
    assert "exact_math_response(" not in music_source
    assert "find_meme_fact(" not in music_source
    assert "emoji_semantic_response(" not in music_source
    assert "self.reliable.answer(" not in music_source
    assert "self._assist_identity(" not in music_source
    assert "self._assist_meme(" not in music_source


def test_music_prompt_teaches_original_copyright_boundary() -> None:
    assert "copyrighted lyrics" in MUSIC_SYSTEM_PROMPT.lower()
    assert "every generated lyric must be new" in MUSIC_SYSTEM_PROMPT.lower()


def test_music_dataset_is_large_and_unique() -> None:
    path = Path(__file__).parents[1] / "data" / "music_v1_conversations.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    serialized = {json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records}
    assert len(records) >= 2_700
    assert len(serialized) >= 2_700
    assert any("funny" in json.dumps(record).lower() or "ridiculous" in json.dumps(record).lower() for record in records)


def test_music_dataset_teaches_complete_song_contract_and_followups() -> None:
    path = Path(__file__).parents[1] / "data" / "music_v1_conversations.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assistant_text = "\n".join(
        message["content"]
        for record in records
        for message in record["messages"]
        if message["role"] == "assistant"
    )
    assert assistant_text.count("Title:") >= 500
    assert assistant_text.count("Style:") >= 500
    assert assistant_text.count("[Verse 1]") >= 500
    assert assistant_text.count("[Chorus]") >= 500
    assert assistant_text.count("[Bridge]") >= 500
    assert assistant_text.count("[Outro]") >= 500
    assert sum(len(record["messages"]) >= 6 for record in records) >= 150


def test_music_candidate_ranking_prefers_complete_generated_song() -> None:
    fragment = "Plastic bending with one flicker and a friendship routine."
    complete = """Title: Signal After Midnight
Style: dark electronic

[Verse 1]
The router blinked and lost the room.
I chased one signal through the gloom.

[Chorus]
Come back online, come back tonight,
Turn every dead bar into light.

[Bridge]
The static drops; the heartbeat stays.

[Outro]
One final light survives the haze."""
    prompt = "Write a full song about my WiFi dying"
    assert MusicModelService._candidate_score(prompt, complete) > MusicModelService._candidate_score(prompt, fragment)


def test_music_candidate_ranking_prefers_title_and_style_for_recall() -> None:
    prompt = "What style and song name did we choose?"
    recalled = "Title: Neon Summer\nStyle: nostalgic synth-pop with warm pads."
    unrelated = "Here is a new chorus about a toaster."
    assert MusicModelService._candidate_score(prompt, recalled) > MusicModelService._candidate_score(prompt, unrelated)
