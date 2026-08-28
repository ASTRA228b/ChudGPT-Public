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


def test_music_topic_relevance_prefers_requested_subject() -> None:
    water = "[Verse 1]\nRiver water rolls toward the ocean.\n[Chorus]\nWaves rise with the tide beside the shore."
    hallway = "[Verse 1]\nA red hallway flickers at night.\n[Chorus]\nThe screen goes black beneath the light."
    prompt = "Write me a full song about water"
    assert MusicModelService._candidate_score(prompt, water) > MusicModelService._candidate_score(prompt, hallway)


def test_music_you_topic_means_chudgpt() -> None:
    assert MusicModelService._topic_relevance(
        "Write me a song about you", "I am ChudGPT, a little model turning prompts into a voice."
    ) > MusicModelService._topic_relevance(
        "Write me a song about you", "The hallway has a red light and an old chair."
    )


def test_music_structure_validator_repairs_labels_and_outro_order() -> None:
    service = object.__new__(MusicModelService)
    service.allowed_sections = {
        "verse 2": "Verse 2", "chorus": "Chorus", "bridge": "Bridge", "outro": "Outro"
    }
    text = "Styp: synth pop\n\n[Verse 2]\nFirst line.\n\n[Outro]\nBye.\n\n[Bridge]\nWait."
    fixed, corrections = service._validate_structure(text)
    assert "Style: synth pop" in fixed
    assert "[Verse 1]" in fixed
    assert fixed.rfind("[Outro]") > fixed.rfind("[Bridge]")
    assert corrections


def test_music_request_source_is_validated() -> None:
    from public_api_server import ChatRequest
    assert ChatRequest(message="song", source="webclient").source == "webclient"


def test_music_ordinary_song_rejects_metadata_only_fragment() -> None:
    broken = "Driving\nStylas: glorious mistake, textured percenturpose."
    lyrics = "[Verse 1]\nMy compiler hums while every tired key keeps time.\n\n[Chorus]\nCode through the night, fix one more line."
    prompt = "Write a song about coding"
    assert MusicModelService._candidate_score(prompt, lyrics) > MusicModelService._candidate_score(prompt, broken)


def test_music_validator_normalizes_stylas() -> None:
    service = object.__new__(MusicModelService)
    service.allowed_sections = {}
    fixed, corrections = service._validate_structure("Driving\nStylas: chaotic rock")
    assert fixed == "Driving\nStyle: chaotic rock"
    assert "normalized-style-label" in corrections


def test_music_validator_removes_memorized_style_suffix_and_static_title_prefix() -> None:
    service = object.__new__(MusicModelService)
    service.allowed_sections = {}
    text = "static Title: Very After Midnight\nStyle: microwave; sad, with a clear pulse and a slightly unwise finale"
    fixed, corrections = service._validate_structure(text)
    assert fixed == "Title: Very After Midnight\nStyle: microwave; sad"
    assert "normalized-title-label" in corrections
    assert "removed-memorized-style-suffix" in corrections


def test_music_removes_only_log_proven_overrepresented_lines() -> None:
    service = object.__new__(MusicModelService)
    service.overrepresented_music_lines = {"the hallway hums in a tired key,"}
    service.overrepresented_music_phrases = set()
    fixed, removed = service._remove_overrepresented_lines(
        "[Verse 1]\nThe hallway hums in a tired key,\nA new coding rhythm wakes at dawn."
    )
    assert removed == 1
    assert "hallway" not in fixed.lower()
    assert "coding rhythm" in fixed


def test_music_removes_paraphrased_log_proven_phrase_family() -> None:
    service = object.__new__(MusicModelService)
    service.overrepresented_music_lines = set()
    service.overrepresented_music_phrases = {"light keeps judging me"}
    fixed, removed = service._remove_overrepresented_lines(
        "[Chorus]\nA little light keeps judging me.\nThe bass wakes up beneath the street.",
        "Make a dark electronic chorus",
    )
    assert removed == 1
    assert "judging" not in fixed.lower()
    assert "bass wakes" in fixed


def test_music_keeps_repeated_phrase_when_user_explicitly_requests_it() -> None:
    service = object.__new__(MusicModelService)
    service.overrepresented_music_lines = set()
    service.overrepresented_music_phrases = {"light keeps judging me"}
    fixed, removed = service._remove_overrepresented_lines(
        "[Chorus]\nThe light keeps judging me.",
        "Write a chorus using the phrase light keeps judging me",
    )
    assert removed == 0
    assert "light keeps judging me" in fixed.lower()


def test_music_repetition_filter_never_erases_the_neural_draft() -> None:
    service = object.__new__(MusicModelService)
    service.allowed_sections = {"chorus": "Chorus"}
    service.overrepresented_music_lines = set()
    service.overrepresented_music_phrases = {"light keeps judging me"}
    neural = "[Chorus]\nA little light keeps judging me."
    filtered, corrections = service._safely_filter_repetition(neural, "Write a dark chorus")
    assert filtered == neural
    assert "repetition-filter-reverted-destructive" in corrections


def test_music_repetition_filter_keeps_a_smaller_novel_draft() -> None:
    service = object.__new__(MusicModelService)
    service.allowed_sections = {"verse 1": "Verse 1", "chorus": "Chorus"}
    service.overrepresented_music_lines = set()
    service.overrepresented_music_phrases = {"light keeps judging me"}
    neural = """[Verse 1]
A little light keeps judging me.
The router rides the thunder while the copper cables sing.

[Chorus]
The storm can shake the windows, but the network holds the line."""
    filtered, corrections = service._safely_filter_repetition(neural, "router thunderstorm song")
    assert "judging" not in filtered.lower()
    assert "router rides" in filtered.lower()
    assert any(item.startswith("removed-overrepresented-lines:") for item in corrections)


def test_write_me_a_song_requires_a_complete_generated_song() -> None:
    fragment = "[Chorus]\nOne lonely line."
    complete = """Title: Network Weather
Style: glitch rock

[Verse 1]
The router shakes awake while every blue light paints a rhythm across the room tonight.

[Chorus]
Carry the signal home through static, thunder, broken cables, and the restless air.

[Bridge]
One final packet finds the road and turns the silence into sound again."""
    assert not MusicModelService._candidate_meets_music_shape("Write me a song", fragment)
    assert MusicModelService._candidate_meets_music_shape("Write me a song", complete)


def test_full_song_shape_rejects_numbered_title_dump() -> None:
    broken = "1. Orbit\n2. Rain A\n3. Style: faded villages\n4. Static"
    assert not MusicModelService._candidate_meets_music_shape("Write me a full song", broken)


def test_full_song_shape_accepts_generated_song_structure() -> None:
    reply = """Title: Neon Rain
Style: dark electronic rock

[Verse 1]
The first generated verse carries enough original words to establish the scene and subject clearly.

[Chorus]
The generated chorus returns with a distinct hook and keeps the requested musical idea moving forward.

[Bridge]
The bridge changes perspective before the final rhythm resolves the song with several more original words."""
    assert MusicModelService._candidate_meets_music_shape("Write me a full song", reply)
