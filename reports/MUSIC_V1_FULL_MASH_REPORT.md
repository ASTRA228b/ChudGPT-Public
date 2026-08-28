# ChudGPT-Public-Music V1 full-song and lyric-mash pass

Date: 2026-08-28

## Outcome

Music V1 now distinguishes full songs, short songs, individual sections, title/style work,
rewrites, continuations, feedback, rhyme help, and lyric mash requests. Full-song requests
that end prematurely receive one second neural composition pass using the same Music V1
checkpoint. Code supplies section labels only; title, style, and new lyric text are sampled
from Music V1. In mash mode, the user's own supplied lines are preserved as source material
and Music V1 generates the surrounding sections.

The existing `checkpoints/public_music_v1/best.pt` remains active. An experimental 150-step
correction run lowered validation loss but produced worse sampled songs, so it was not
promoted.

## Root causes fixed

- Intent/full-song instructions were previously appended after context tokenization, so the
  model never saw them.
- The runtime treated many full-song and mash fragments as acceptable outputs.
- The repetition cleaner could destroy an otherwise complete song after validation.
- Mash validation examined only the final instruction and ignored lyric scraps in earlier
  conversation turns.
- The corpus underrepresented tiny songs, continuation follow-ups, and lyric mash tasks.

## Dataset

- Before this pass: 3,970 unique conversations.
- After this pass: 4,330 unique conversations (4,330/4,330 unique).
- Multi-turn conversations: 891.
- Six-turn conversations: 230.
- Explicit lyric-mash conversations: 230.
- Explicit tiny four-line song conversations: 241.
- Explicit second-verse continuation conversations: 201.
- Duplicate output records after generation/deduplication: 0.

The previous corpus was backed up under
`data/archive/music_v1_pre_full_mash_20260827/` locally before rebuilding.

## Training experiment

The full/mash corpus was fine-tuned from the active checkpoint on CUDA for 150 steps.

- Step 50 validation loss: 4.4356
- Step 100 validation loss: 4.2061
- Step 150 validation loss: 4.1415
- NaN/OOM failures: 0

Despite lower validation loss, direct generation quality regressed (short fragments,
malformed metadata, and weaker topic relevance). The checkpoint is retained only as a local
experiment in `checkpoints/public_music_v1_full_mash/`; production continues to use the
stronger prior checkpoint.

## Before/after live checks

### Full song

- Before: often 1-3 sections or a title/style fragment.
- After: the tested CUDA request produced 202 words and six labeled lyric sections through
  the second neural composition pass.
- Remaining weakness: individual lines are often grammatically broken and topic adherence is
  weak because the active model is only about 21M parameters.

### Lyric mash

- Before: 27 unrelated words; neither supplied scrap survived.
- After: 62 words and five labeled sections; both `blue paper satellites` and
  `coffee on the dashboard` were retained verbatim as user-owned source lines.
- Remaining weakness: Music V1's newly generated connecting lines remain sloshy.

### Short song

The tested short request stayed short (22 words, one section) rather than being inflated into
a full song.

## Verification

- Music V1 focused tests: 35 passed.
- Complete ChudGPT-Public test suite: 311 passed.
- Dataset uniqueness: 4,330 unique of 4,330 records.
- CUDA generation exercised: full song, short song, and multi-turn lyric mash.

## Known limitations

This pass makes output length, structure, task selection, and lyric-source preservation more
reliable. It cannot turn a 21M-parameter checkpoint into a large songwriting model. Grammar,
coherence, rhyming, metadata quality, and long-range thematic development remain limited.
