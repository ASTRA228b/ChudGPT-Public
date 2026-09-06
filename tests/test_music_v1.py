from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from music_instructions import MUSIC_MODEL_NAME, MUSIC_SYSTEM_PROMPT
from public_api_server import MusicModelService
from build_music_v1_data import _normalized_training_lines, decontaminate
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
    assert "canned_greeting_response(" not in music_source
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
    assert len(records) >= 3_300
    assert len(serialized) >= 3_300
    assert any("funny" in json.dumps(record).lower() or "ridiculous" in json.dumps(record).lower() for record in records)

    line_counts: dict[str, int] = {}
    for record in records:
        assistant_text = "\n".join(
            message["content"] for message in record["messages"]
            if message["role"] == "assistant"
        )
        for line, occurrences in _normalized_training_lines(assistant_text).items():
            line_counts[line] = line_counts.get(line, 0) + occurrences
    assert max(line_counts.values()) <= 6


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
    assert assistant_text.count("[Bridge]") >= 300
    assert assistant_text.count("[Outro]") >= 400
    assert assistant_text.count("[Intro]") >= 250
    assert assistant_text.count("[Pre-Chorus]") >= 100
    assert assistant_text.count("[Instrumental Break]") >= 100
    assert sum(len(record["messages"]) >= 6 for record in records) >= 100
    serialized_records = "\n".join(json.dumps(record, ensure_ascii=False).lower() for record in records)
    assert serialized_records.count("mash those lyrics together") >= 100
    assert serialized_records.count("tiny 4-line song") >= 150
    assert serialized_records.count("continue it with a second verse") >= 150


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
    assert 'detected_intent == "FULL_SONG"' in source
    assert "minimum_draft_tokens = min(220, requested_tokens - 1)" in source


def test_music_intent_classifier_distinguishes_songwriting_jobs() -> None:
    cases = {
        "Write me a full song about coding at 3 AM": ("FULL_SONG", "full"),
        "Make a tiny funny song about a microwave": ("SHORT_SONG", "short"),
        "Give me a chorus about feeling lost": ("CHORUS", "focused"),
        "Give me 5 hook ideas": ("HOOK", "focused"),
        "What rhymes with crash?": ("RHYME_HELP", "focused"),
        "Rewrite this lyric to sound darker": ("REWRITE_LYRIC", "focused"),
        "Continue these lyrics: the rain came down": ("CONTINUE_LYRICS", "focused"),
        "Give me ten song title ideas about computers": ("TITLE_IDEAS", "focused"),
        "Give me five styles for a dark electronic song": ("STYLE_IDEAS", "focused"),
        "Mash these lyrics together into one song": ("MASH_LYRICS", "full"),
    }
    for prompt, expected in cases.items():
        profile = MusicModelService._classify_music_request(prompt)
        assert (profile["intent"], profile["requested_length"]) == expected


def test_music_intent_guidance_contains_no_canned_song_content() -> None:
    full = MusicModelService._music_generation_instruction("FULL_SONG", "full")
    mash = MusicModelService._music_generation_instruction("MASH_LYRICS", "full")
    assert "at least five labeled lyric sections" in full
    assert "supplied in this conversation" in mash
    assert "hallway" not in full.lower() + mash.lower()
    assert "midnight" not in full.lower() + mash.lower()


def test_music_full_short_and_partial_shapes_are_different() -> None:
    full = """Title: Rain Engine
Style: energetic rock with heavy drums

[Intro]
Clouds gather over the avenue and the first drops mark the road.
[Verse 1]
Water runs from every roof while I keep walking through the weather.
[Pre-Chorus]
Every silver puddle starts to shake beneath the growing thunder.
[Chorus]
Let the rain come down, let the whole town sing its name together.
[Verse 2]
Rivers form beside my shoes and carry yesterday toward the sea.
[Bridge]
For one quiet breath the storm releases everything it held.
[Final Chorus]
Let the rain come down, let the whole town sing its name together.
[Outro]
Morning finds the street still shining after all the clouds have gone."""
    short = "[Verse 1]\nRain taps twice against my door.\n\n[Chorus]\nCome down, rain, then ask for more."
    titles = "1. Weather Without Walls\n2. Silver Street\n3. After the Downpour"
    assert MusicModelService._candidate_meets_music_shape("Write a full song about rain", full)
    assert not MusicModelService._candidate_meets_music_shape("Write a short song about rain", full)
    assert MusicModelService._candidate_meets_music_shape("Write a short song about rain", short)
    assert MusicModelService._candidate_meets_music_shape("Give me song title ideas about rain", titles)


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


def test_music_relevance_requires_combined_subject_not_one_generic_alias() -> None:
    prompt = "Write me a full song about a robot learning to garden"
    generic = "A machine starts dancing in a hallway beneath a light."
    relevant = "A robot learns to garden, planting seeds in careful rows."
    assert MusicModelService._topic_relevance(prompt, generic) < 0.5
    assert MusicModelService._topic_relevance(prompt, relevant) >= 0.5


def test_music_training_decontamination_caps_repeated_lines_and_titles() -> None:
    rows = [
        {"messages": [
            {"role": "user", "content": f"song {index}"},
            {"role": "assistant", "content": (
                "Title: Same Name\n[Verse]\nThe same memorized lyric line appears right here.\n"
                f"A distinct generated training line carries number {index}."
            )},
        ]}
        for index in range(8)
    ]
    kept = decontaminate(rows, line_cap=2, title_cap=3)
    assert len(kept) == 2


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


def test_music_validator_removes_memorized_suffix_outside_style_metadata() -> None:
    service = object.__new__(MusicModelService)
    service.allowed_sections = {"bridge": "Bridge"}
    fixed, corrections = service._validate_structure(
        "[Bridge]\nStywichard; sad, with a clear pulse and a slightly unwise finale\nA novel line remains."
    )
    assert "slightly unwise finale" not in fixed.lower()
    assert "Stywichard; sad" in fixed
    assert "A novel line remains" in fixed
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
I trace the copper cables through the dust beneath the desk and listen for a sign.

[Pre-Chorus]
Every packet waits beside the gate while thunder shakes the window frame.

[Chorus]
Carry the signal home through static, thunder, broken cables, and the restless air.
Carry the signal home until the sleeping network learns my name.

[Verse 2]
The modem blinks a stubborn code; I reset every switch and start the search again.

[Bridge]
One final packet finds the road and turns the silence into sound again.

[Outro]
Morning reaches through the blinds as every quiet status light turns green."""
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
Rain starts tapping on the windows while the whole street waits beneath the clouds.

[Pre-Chorus]
Every silver gutter fills and every distant roof begins to sing aloud.

[Chorus]
The generated chorus returns with a distinct hook and keeps the requested musical idea moving forward.
Let the neon rain keep falling while we turn the restless weather into sound.

[Verse 2]
Puddles catch the traffic lights and scatter every color on the shining ground.

[Bridge]
The bridge changes perspective before the final rhythm resolves the song with several more original words.

[Outro]
Morning clears the avenue and leaves one final drop to mark the closing beat."""
    assert MusicModelService._candidate_meets_music_shape("Write me a full song", reply)


def test_neural_song_assembly_uses_generated_parts_without_canned_lyrics(monkeypatch) -> None:
    service = object.__new__(MusicModelService)
    service.allowed_sections = {
        "verse 1": "Verse 1",
        "intro": "Intro",
        "pre-chorus": "Pre-Chorus",
        "hook": "Hook",
        "chorus": "Chorus",
        "verse 2": "Verse 2",
        "bridge": "Bridge",
        "final chorus": "Final Chorus",
        "outro": "Outro",
    }
    section_parts = [
        "The compiler wakes and throws its sparks across the room.\nI follow every warning while the cooling fans begin to bloom.",
        "The cursor holds its breath before the build begins again.\nA quiet error turns into a rhythm in the rain.",
        "Compile the night and carry every broken line along.\nTurn the red diagnostics into one electric song.",
        "The second build is cleaner but the tests still shake the floor.\nI patch another function and then ask the code for more.",
        "The stack trace twists sideways and reveals a hidden door.\nI change the old assumption that was breaking us before.",
        "Compile the night; the final run is brighter than before.\nEvery passing test becomes a heartbeat through the floor.",
        "The terminal grows quiet as the sunrise finds the screen.\nI save the final changes and the status light turns green.",
    ]
    generated_parts = iter([
        "Title: Copper Weather\nStyle: glitch rock with restless drums",
        "Title: Copper Weather\nStyle: glitch rock with restless drums",
        "Title: Copper Weather\nStyle: glitch rock with restless drums",
        *(part for part in section_parts for _ in range(2)),
    ])

    monkeypatch.setattr(service, "_sample_neural_piece", lambda *args, **kwargs: next(generated_parts))
    assembled, corrections = service._assemble_neural_song(
        [{"role": "user", "content": "Write a full song about coding"}],
        "music system",
        "Write a full song about coding",
        "FULL_SONG",
    )

    assert "Title: Copper Weather" in assembled
    assert "Style: glitch rock" in assembled
    assert len(re.findall(r"(?m)^\[[^]]+\]$", assembled)) == 7
    assert "compiler wakes" in assembled.lower()
    assert "neural-section-assembly" in corrections
    assert MusicModelService._candidate_meets_music_shape("Write a full song about coding", assembled)


def test_mash_intent_requests_preservation_from_neural_model(monkeypatch) -> None:
    service = object.__new__(MusicModelService)
    service.allowed_sections = {
        "verse 1": "Verse 1", "chorus": "Chorus", "verse 2": "Verse 2",
        "breakdown": "Breakdown", "bridge": "Bridge",
        "final chorus": "Final Chorus", "outro": "Outro",
    }
    instructions: list[str] = []
    parts = iter([
        "Title: Paper Satellites\nStyle: crooked synth pop",
        "Title: Paper Satellites\nStyle: crooked synth pop",
        "Title: Paper Satellites\nStyle: crooked synth pop",
        *["Blue paper satellites cross the kitchen sky.\nBorrowed lines return with new connections as they fly."] * 14,
    ])

    def sample(*args, **kwargs):
        instructions.append(args[2])
        return next(parts)

    monkeypatch.setattr(service, "_sample_neural_piece", sample)
    assembled, _ = service._assemble_neural_song(
        [{"role": "user", "content": "Mash these lyrics: blue paper / kitchen sky"}],
        "music system",
        "Mash these lyrics: blue paper / kitchen sky",
        "MASH_LYRICS",
    )

    assert assembled
    assert any("Preserve and recombine" in instruction for instruction in instructions)
    assert "blue paper" in assembled.lower()


def test_mash_shape_requires_user_lyric_material() -> None:
    unrelated = "[Verse 1]\nA hallway machine wanders beneath a distant moon with several unrelated words."
    preserved = """[Verse 1]
Blue paper satellites cross the kitchen sky while coffee waits on the dashboard tonight.
The engine keeps a crooked rhythm and the morning traffic starts to glow.
[Chorus]
Carry every folded signal through the city; let the dashboard coffee overflow."""
    preserved += """
[Verse 2]
Traffic writes another melody while every folded satellite keeps moving.
[Bridge]
Coffee cools beside the wheel as morning opens up the road.
[Outro]
Blue paper settles on the seat and leaves the dashboard glowing."""
    prompt = "Mash these lyrics together: blue paper satellites / coffee on the dashboard"
    assert not MusicModelService._candidate_meets_music_shape(prompt, unrelated)
    assert MusicModelService._candidate_meets_music_shape(prompt, preserved)


def test_mash_source_content_preserves_only_user_supplied_material() -> None:
    history = [
        {"role": "user", "content": "My lyric scraps are: blue paper satellites / coffee on the dashboard"},
        {"role": "assistant", "content": "What should I do with them?"},
        {"role": "user", "content": "Mash those lyrics together"},
    ]
    assert MusicModelService._mash_source_content(history) == (
        "blue paper satellites\ncoffee on the dashboard"
    )


def test_repetition_filter_preserves_a_complete_song_shape(monkeypatch) -> None:
    service = object.__new__(MusicModelService)
    original = """Title: Copper Weather
Style: glitch rock

[Verse 1]
One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen.
[Pre-Chorus]
One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen.
[Chorus]
One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen.
[Verse 2]
One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen.
[Bridge]
One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen.
[Outro]
One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen."""
    damaged = "Title: Copper Weather\nStyle: glitch rock\n\n[Verse 1]\nOnly one line remains."
    monkeypatch.setattr(service, "_remove_overrepresented_lines", lambda reply, prompt: (damaged, 6))
    filtered, corrections = service._safely_filter_repetition(
        original, "Write a full song"
    )
    assert filtered.lower().count("one two three") == 1
    assert corrections == [
        "removed-intra-song-repetitions:5",
        "overrepresented-filter-reverted-shape-loss",
    ]


def test_intra_song_filter_removes_exact_and_close_repeated_lines() -> None:
    reply = """[Verse 1]
The hallway hums in a tired key.
I made one promise to make it through.
[Chorus]
The hallway hums in a tired key.
I made one promise that I would make it through.
The final window opens into morning."""
    filtered, removed = MusicModelService._remove_intra_song_repetition(reply)
    assert removed == 2
    assert filtered.lower().count("hallway hums") == 1
    assert filtered.lower().count("made one promise") == 1
    assert "final window" in filtered.lower()


def test_structure_cleanup_preserves_punctuation_and_drops_empty_sections() -> None:
    service = object.__new__(MusicModelService)
    service.allowed_sections = {"verse 1": "Verse 1", "chorus": "Chorus", "outro": "Outro"}
    fixed, corrections = service._validate_structure(
        "Title: WiFi Funeral\nStyle: glitch pop.\n\n[Verse 1]\nThe router blinks.\n\n[Chorus]\n\n[Outro]\nSignal gone."
    )
    cleaned = service._remove_empty_sections(fixed)
    assert "Style: glitch pop." in cleaned
    assert "[Chorus]" not in cleaned
    assert "removed-memorized-style-suffix" not in corrections


def test_full_song_shape_rejects_empty_or_off_topic_sections() -> None:
    malformed = """Title: Hallway
Style: dark pop

[Verse 1]
The hallway light keeps judging me in tired static.
[Chorus]

[Verse 2]
The hallway stays blue beneath a moon.
[Bridge]
The corridor bends behind an empty door.
[Final Chorus]
The ceiling fan continues through the night.
[Outro]
The room goes black and quiet at dawn.
"""
    assert not MusicModelService._candidate_meets_music_shape(
        "Write a full song about my WiFi dying", malformed
    )
