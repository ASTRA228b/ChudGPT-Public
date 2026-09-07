# Public V20 balanced fine-tuning

## Scope

Model training and evaluation only. No fallback, inference router, Music model,
Discord command, or UI behavior was edited in this pass.

## Training work

- Added an initial 307 reviewed examples combining everyday conversation, contextual
  follow-ups, corrections, Python/Java/JavaScript/C#/SQL, explanations, summaries,
  extraction, and general knowledge.
- Expanded this to a separate 889-example curriculum with varied names,
  multi-turn recall, extraction, and Python/Java variable binding. Both versions
  are reproducible with `build_public_balanced_sft.py` (use `--binding` for 889).
- Multi-turn examples include each assistant-turn prefix so the model learns
  intermediate replies as well as the last answer.
- Fixed validation splitting to keep related conversation prefixes together.
  Previously one prefix could be training data while a longer version containing
  the same answer appeared in validation.
- Kept the new evaluation prompts out of training; a test checks exact overlap.
- Avoided recycling unreviewed source data found to contain false claims and
  replies that require missing conversation context.

## Evaluation

`benchmark_balanced_public.py` contains 22 unseen scenarios: 16 narrow automatic
content/format checks and six conversation/creativity cases requiring manual
review. Every evaluated answer comes from neural generation, bypassing response
routers. Fixed random seeds and identical inference settings are used.

Automatic checks are not a general-intelligence score. For example, the old
checkpoint's numerical pass contained the expected number but an irrelevant
explanation, so manual reading remains necessary.

| Checkpoint | Automatic checks |
| --- | --- |
| Previously served V20 | 1/16 |
| Reviewed curriculum, 240 steps | 0/16 |
| Additional 720-step fine-tune | 1/16 |
| Recovery from earlier V20, 900 steps | 0/16 |
| Expanded binding curriculum, 900 steps | 0/16 |

The initial candidates still mix training fragments and fail to follow many
requests. Lower validation loss did not demonstrate acceptable conversational
or coding quality. Full local outputs are in `reports/balanced_benchmark_*.json`.

Software verification: 365 tests passed, including new checks for conversation
split isolation and evaluation/training separation.

## Confirmation and deployment decision

A separate ten-scenario confirmation suite adds eight automatic checks and two
conversation cases. The served checkpoint and expanded candidate both scored
0/8. Disabling repetition penalties for an isolated candidate evaluation also
scored 0/8; production decoding was therefore left unchanged.

There are 32 new benchmark scenarios in total. The automatic checks are stricter
than the older broad benchmark, so scores from those suites are not comparable.
Manual review found persistent irrelevant replies, invented names, and code that
does not match the requested operation or identifiers. The candidates sometimes
improved syntax and individual facts, but not reliable task completion.

No candidate met the release standard. The existing served checkpoint remains
selected. All training code, datasets, configurations, tests, and evaluations are
saved; candidate weights remain local under `checkpoints/`. No fallback source
or inference behavior changed. This pass completed actual fine-tuning experiments
but did not establish a smarter release. Further work needs substantially better
generalization, rather than treating lower teacher-forced loss as proof of quality.
