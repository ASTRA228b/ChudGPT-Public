# ChudGPT-Public-Music V1: final quality and reliability report

## 1. What was wrong

The active Music checkpoint is overfit to a small set of room/static/signal lyrics. It often repeats "After Midnight," "the room stays blue," "name it courage," and the same signal/screen outro. The prior synthetic corpus looked large but was assembled from only 20 fixed verse pairs, 12 fixed chorus pairs, and a few bridge/outro blocks. Its old subject-splicing rule also taught malformed sentences such as "As microwave running for president drifts past...".

## 2. What the logs showed

Private Music/Discord logs reproduced the same stock lines, malformed titles, prompt drift, and weak grammar. The analyzer reads these private JSONL files locally; no private log is published. It reports repeated lines and 3-6 word phrases, title/style duplication, output length, internal repetition, and malformed sections.

## 3. Repetition findings

The first corpus had five full lyric lines occurring 1,076 times each. The final rebuilt corpus has 3,600 unique conversations and 4,320 assistant turns. Its highest ordinary topic-bearing lyric repetition is five; the remaining 40-use entries are short instrumental-break directions. It contains 506 unique titles and 1,252 unique style descriptions. The active checkpoint remains contaminated by its older training history, so a clean corpus alone cannot repair it without a successful replacement checkpoint.

## 4. Dataset changes

- Added broader genre, mood, subject, tempo, texture, vocal, and arrangement coverage.
- Expanded verse/chorus writing and independently combines compatible first/second lines instead of reusing fixed pairs.
- Removed the malformed "As <whole subject> drifts past" construction.
- Added grammatical topic-aware bridge/outro frames so stock lines are not copied unchanged across hundreds of songs.
- Preserved request scope: ordinary song drafts omit title/style; explicit title/style requests generate them; complete-song requests can use longer structures.
- The pre-change corpus remains backed up under `data/archive/music_v1_pre_intelligent_slop_20260827/`.

## 5. Training changes

Training now starts from the compatible `public_v20_quality` checkpoint rather than compounding the overfit Music checkpoint. The same 20,999,184-parameter architecture and tokenizer are retained. Two CUDA experiments were run: 500 steps on the first clean rebuild (validation loss 2.3427), then 1,000 steps on the expanded rebuild (validation loss 1.3435). Neither produced better human-readable unseen songs, so neither was promoted.

## 6. Generation changes

Music still uses six neural samples, prompt-aware candidate ranking, four-token no-repeat n-grams, session title/style/line similarity penalties, and no canned song fallback. A lower-temperature decoding sweep was tested and rejected: on the active checkpoint it increased cross-output repeated lines from 55 to 113. The proven active decoding settings were restored.

## 7. Discord behavior

`!chud music <prompt>` still routes only to Music V1. Long output uses the existing Discord-safe section-aware splitter rather than cutting a lyric line in half. Normal Public and unrelated commands were not changed.

## 8. Exact non-cherry-picked benchmark

Ten requested prompts were each sampled twice. With the original decoder, the active checkpoint scored: prompt overlap 0.021, 55 cross-output repeated lines, 0 internal repeated lines, and 0 malformed-output flags. The 1,000-step fresh candidate scored: prompt overlap 0.017, 0 cross-output repeated lines, 0 internal repeated lines, and 0 malformed-output flags, but qualitative grammar and topic adherence were materially worse. The candidate was therefore rejected. Complete raw samples are retained in the local benchmark JSON files.

## 9. Before/after examples and verdict

Before: the active checkpoint produced recognizable sections but recycled "After Midnight," "the room stays blue," and "I laugh once softly and don't look back." Candidate: copied lines disappeared, but samples such as "The old mall barefternoon..." were less readable. Final verdict: keep the active checkpoint and decoder; ship the corrected corpus, evaluation, and analysis pipeline; archive both rejected candidates. This is not claimed as a neural-quality win.

## 10. Files changed

- `build_music_v1_data.py`
- `data/music_v1_conversations.jsonl`
- `configs/finetune_music_v1.yaml`
- `evaluate_music_v1.py`
- `reports/MUSIC_V1_INTELLIGENT_SLOP_REPORT.md`

Known limitation: a 21M decoder can learn the format and vocabulary yet still fail semantic composition. Lower validation loss did not guarantee better lyrics. A future promotion must beat the live checkpoint in blind human review as well as repetition metrics.
