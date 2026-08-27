from __future__ import annotations

import json
from pathlib import Path

import torch

from music_instructions import MUSIC_MODEL_NAME, MUSIC_SYSTEM_PROMPT
from public_api_server import MusicModelService
from chudlm.generation import generate


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
    assert assistant_text.count("[Bridge]") >= 400
    assert assistant_text.count("[Outro]") >= 400
    assert assistant_text.count("[Intro]") >= 300
    assert assistant_text.count("[Pre-Chorus]") >= 100
    assert assistant_text.count("[Instrumental Break]") >= 100
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


def test_music_metadata_is_preferred_only_when_requested() -> None:
    lyrics_only = """[Verse 1]\nWe met beneath the station light.\n\n[Chorus]\nStay with me through the night.\n\n[Bridge]\nThe city fades from view.\n\n[Outro]\nI still come home to you."""
    with_metadata = "Title: Station Light\nStyle: synth-pop\n\n" + lyrics_only
    assert MusicModelService._candidate_score("Make a love song", lyrics_only) > MusicModelService._candidate_score(
        "Make a love song", with_metadata
    )
    assert MusicModelService._candidate_score(
        "Make a love song with a title and synth-pop style", with_metadata
    ) > MusicModelService._candidate_score("Make a love song with a title and synth-pop style", lyrics_only)


def test_music_full_length_is_only_for_explicit_full_requests() -> None:
    source = (Path(__file__).parents[1] / "public_api_server.py").read_text(encoding="utf-8")
    assert "wants_complete_song" in source
    assert "desired_sections = 3 if wants_complete_song else 2" in source
    assert "minimum_draft_tokens = min(140, requested_tokens - 1) if wants_complete_song else 0" in source


def test_music_candidate_ranking_prefers_title_and_style_for_recall() -> None:
    prompt = "What style and song name did we choose?"
    recalled = "Title: Neon Summer\nStyle: nostalgic synth-pop with warm pads."
    unrelated = "Here is a new chorus about a toaster."
    assert MusicModelService._candidate_score(prompt, recalled) > MusicModelService._candidate_score(prompt, unrelated)


def test_decoder_can_delay_eos_without_inserting_answer_text() -> None:
    class EosFirstModel:
        class Config:
            context_length = 16

        config = Config()

        def __call__(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, None]:
            logits = torch.zeros((1, token_ids.shape[1], 4))
            logits[:, -1, 2] = 10.0  # EOS is always the preferred token.
            logits[:, -1, 1] = 9.0
            return logits, None

    output = generate(
        EosFirstModel(),
        torch.tensor([[0]]),
        max_new_tokens=6,
        temperature=0,
        eos_token_id=2,
        min_new_tokens=3,
    )
    # Three model-selected non-EOS tokens are followed by the model's EOS.
    assert output.tolist() == [[0, 1, 1, 1, 2]]


def test_decoder_blocks_repeated_four_grams_without_inserting_text() -> None:
    class RepeatModel:
        class Config:
            context_length = 32

        config = Config()

        def __call__(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, None]:
            logits = torch.zeros((1, token_ids.shape[1], 6))
            preferred = [1, 2, 3, 4][(token_ids.shape[1] - 1) % 4]
            logits[:, -1, preferred] = 10.0
            logits[:, -1, 5] = 9.0
            return logits, None

    output = generate(RepeatModel(), torch.tensor([[0]]), max_new_tokens=9, temperature=0,
                      no_repeat_ngram_size=4)
    generated = output.tolist()[0][1:]
    four_grams = [tuple(generated[index:index + 4]) for index in range(len(generated) - 3)]
    assert len(four_grams) == len(set(four_grams))


def test_music_candidate_score_rejects_degenerate_titles() -> None:
    bad = "Title: A A A\nStyle: dark pop with soft drums\n\n[Verse]\nNothing waits beside the door."
    good = "Title: The Quiet Department\nStyle: dark pop with soft drums\n\n[Verse]\nNothing waits beside the door."
    prompt = "Give me a title and style for a song about nothing"
    assert MusicModelService._candidate_score(prompt, good) > MusicModelService._candidate_score(prompt, bad)
