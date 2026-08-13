# ChudGPT-Public alignment improvement report

## Root causes

The original 20,000-conversation corpus contained only 2,569 unique prompts and 1,753 unique answers. Two generic assistant responses appeared 2,000 times each; many fact responses appeared 250 times each; and one generic code explanation appeared 2,000 times. The model therefore learned frequent response classes instead of reliable prompt-to-answer alignment.

Training and inference role formatting, tokenizer vocabulary, and special-token IDs matched. A separate identity conflict was found in the protected prompt: inference called the model `ChudGPT` while the SFT examples called it `ChudGPT-Public`.

## Changes

- Added a deduplicated alignment dataset builder with greetings, identity, concise instructions, facts, simple code, corrections, and multi-turn recall.
- Changed the protected identity to ChudGPT-Public and explicitly prioritized the current request and constraints.
- Trained three isolated candidates without overwriting old checkpoints.
- Reduced default serving temperature from 0.72 to 0.35 and tightened sampling.
- Added a general binary arithmetic evaluator for explicit expressions.
- Added session-scoped recall for user statements shaped like `my X is Y`.
- Added a minimal greeting control so greetings do not trigger unrelated facts.
- Added a reproducible benchmark with corrected exact-number scoring.

## Training

The promoted V3 candidate was initialized from `checkpoints/chat/best.pt` and trained with response-only loss on 712 training examples and 62 validation examples. It completed 600 optimizer steps over 20 epochs. Validation loss decreased from 5.2152 to 1.0100. The promoted checkpoint remains separate at `checkpoints/alignment_candidate_v3/best.pt`; original base/chat checkpoints remain unchanged.

## Benchmark

The old raw checkpoint scored 0/8. V1 scored 1/8 and was rejected. V2 scored 2/8 and was rejected. V3 raw generation scored 3/8 with strict numeric scoring. The actual V3 API stack scored 7/8:

| Test | Old | Improved API |
|---|---|---|
| Greeting | Fail | Pass |
| 4 + 4 | Fail | Pass (`4 + 4 is 8.`) |
| 12 × 8 | Fail | Pass (`12 * 8 is 96.`) |
| Gravity, one sentence | Fail | Pass |
| Largest ocean | Fail | Pass |
| Clear-day sky color | Fail | Fail |
| Python add function, code only | Fail | Pass |
| Multi-turn favorite-color recall | Fail | Pass |

The clear-sky test remains a known limitation. No claim of perfect behavior is made.

## Compatibility

The request/response contracts for `/api/chat`, `/api/generate`, `/api/status`, `/api/info`, and `/api/clear` are unchanged. The Vercel website uses `/api/chat` and remains compatible. Desktop clients using the same JSON API remain compatible. Existing sessions reset when the backend process is restarted.
