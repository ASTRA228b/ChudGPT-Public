# ChudGPT-Public-Music V1: Intelligent Slop Pass

## 1. What was wrong

The original Music V1 corpus was narrow and heavily repeated. Five complete lyric lines occurred 1,076 times each, "screen goes black" appeared across 1,166 outputs, and generated songs frequently reused the same room/static/signal language. The live Discord Music samples also showed malformed titles, disconnected sections, prompt drift, and repeated old outros.

## 2. Data and log findings

- Old corpus: 3,358 assistant outputs, 80 unique titles, and 464 unique styles.
- Rebuilt corpus: 3,600 unique conversations, 4,320 assistant outputs, 506 unique titles, and 1,252 unique styles.
- The worst ordinary lyric line dropped from 1,076 occurrences to 10. Forty-use lines in the rebuilt corpus are short instrumental directions rather than song lyrics.
- The private Discord JSONL log contained six Music V1 exchanges that reproduced the same stock-line and malformed-language failures.

## 3. Dataset changes

The corpus builder now combines 51 subjects, 30 genres, 10 moods, varied rhythmic/texture/vocal/arrangement descriptions, multiple section layouts, request-scoped fragments, full songs, title/style-only requests, and multi-turn revisions. The previous corpus is backed up under `data/archive/music_v1_pre_intelligent_slop_20260827/`.

## 4. Runtime changes

- Added configurable no-repeat n-gram decoding; Music V1 uses a four-token window.
- Uses six fixed neural sampling profiles, repetition penalty 1.16, and prompt/repetition-aware candidate ranking.
- Penalizes repeated prior-session titles, styles, and lyric lines while allowing explicit continuity requests.
- Rejects degenerate one-token/repeated-word titles and underspecified styles during candidate ranking.
- No completed song, title, style, or fallback answer is inserted by runtime code.

## 5. Private diagnostics

Each Music generation writes private JSONL diagnostics with UTC timestamp, session, prompt, complete neural reply, checkpoint, chosen title/style, sections, length, score, decoding profiles, and repetition indicators. The analyzer reports corpus and private Discord repetition without exposing private logs publicly.

## 6. Retraining result

The same 20,999,184-parameter architecture was fine-tuned for 600 optimizer steps on CUDA. Validation loss fell from 4.1571 at step 20 to 1.2428 at step 600 without NaNs. However, blind generation review showed that the final and intermediate checkpoints were less grammatical and less prompt-faithful than the proven active checkpoint. They were therefore archived as experiments and not promoted. This is an explicit quality decision: lower validation loss did not equal better conversation.

## 7. Non-cherry-picked benchmark

The archived active checkpoint and candidate were each tested on the same ten unseen prompts with two samples each. The candidate improved unique-title ratio from 0.625 to 1.0 and reduced cross-output repeated lines from 58 to 0, but mean prompt-content overlap fell from 0.105 to 0.081 and qualitative coherence regressed. Every generated sample is retained in the JSON benchmark reports locally.

## 8. Discord behavior

`!chud music` remains routed only to the Music V1 endpoint. Its maximum neural output is now 400 tokens, and existing Discord-safe message splitting preserves long lyrics across multiple messages. Other bot commands and normal Public routing are unchanged.

## 9. Tests

Focused Music/Public regression suite: 209 passed. New coverage checks request scope, title/style diversity behavior, degenerate-title scoring, varied section layouts, decoder n-gram blocking, and the absence of canned runtime songs.

## 10. Honest limitation

Music V1 is still a 21M-parameter experimental model. It can be funny, strange, and more varied, but it cannot reliably match a large production model's long-range lyrical coherence. The rejected checkpoints are kept for research; the deployed checkpoint remains the most coherent tested option.
