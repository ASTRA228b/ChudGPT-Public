# ChudGPT-Public v10 improvement report

## Verdict

Public was drifting into Buggy-style nonsense because its corpus was imbalanced, generic assistant patterns were repeated, and the 21M model did not have enough capacity or clean coverage to generalize reliably. The former v9 live checkpoint scored **7/20 raw** on the new unseen mixed suite and **0/8** on the strict fixed benchmark.

The selected v10-balanced checkpoint scores **8/20 raw** and **12/20 production** with narrowly scoped identity/family assistance. The strict fixed benchmark remains **0/8**, so this is an incremental repair, not a claim that the model is dependable.

## Root causes

- Public v9 contained 3,601 math-like conversations but only about 20 identity and 5 AI-concept conversations.
- The first v10 builder requested hundreds of identity rows, but deduplication collapsed them to roughly 97 distinct examples.
- Repeated caption, joke, math, and generic-topic patterns contaminated unrelated short prompts.
- Continuing from v9 produced only 5/20 raw, evidence of regression/catastrophic forgetting.
- The tokenizer retains legacy malformed-Unicode artifacts. Replacing it would invalidate every checkpoint, so it was not changed silently.

## Dataset

- `data/public_v10_conversations.jsonl`: **12,000 conversations, all unique**.
- **432 identity/family**, **276 AI-concept**, and about **742 meme/slang** examples with varied wording.
- Math-like rows reduced from 3,601 in v9 to **1,291**.
- Removed known malformed-Unicode forms and overused books, speed-times-time, and price templates.
- Capped exact assistant-response repetition.

## Training

- Architecture unchanged: 20,999,184 parameters, 9 layers, width 384, 6 heads, 1,808 FFN, 1,024-token context.
- Response-only SFT from `checkpoints/public_v8/best.pt`.
- CUDA + AMP; batch 6; gradient accumulation 4; LR 2.5e-5; 800 steps.
- Best validation loss: **1.6730**.
- Rejected v9-base candidate: 450 steps, 1.7254 validation loss, 5/20 raw.
- Rejected first clean candidate: 600 steps, 1.7693 validation loss.
- Selected: `checkpoints/public_v10_balanced/best.pt`.

## Runtime

- The neural model always generates first.
- Four neural candidates are ranked for readability and topical overlap; the scorer never inserts a normal conversational answer.
- No generic canned fallback exists.
- Controlled assistance covers only stable Public identity and ChudGPT-family metadata and is disclosed through `assistance_used` and `assistance_reason`.
- `--disable-assistance` provides honest raw-model mode.

## Results

| Evaluation | Previous v9 | Selected v10 |
|---|---:|---:|
| Unseen mixed suite, raw | 7/20 | 8/20 |
| Unseen mixed suite, production | not audited with v10 rules | 12/20 |
| Strict fixed benchmark | 0/8 | 0/8 |
| Short/adversarial suite | 19/40 | 20/40 |
| Unit/regression tests | — | 10/10 |

## Remaining weaknesses

Raw Public still emits malformed fragments, fails arithmetic, loses references, and may produce the wrong coding language. A 21M model trained from scratch on this corpus cannot approach modern assistant quality. The next meaningful leap requires a much larger clean pretraining corpus, tokenizer rebuild, full retraining, and checkpoint selection against unseen conversations—not more keyword rules.
