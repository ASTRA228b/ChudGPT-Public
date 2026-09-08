# ChudGPT-Public V20 retraining report — 2026-09-08

## Outcome

The production checkpoint remains `checkpoints/public_v20_challenge_polish/latest.pt`.
It passed all 10 established capability challenges again. None of the new
broad-training candidates was promoted because they were less reliable and
frequently mixed fragments of unrelated answers into new prompts.

## Work completed

- Added reproducible curriculum builders for K–5, middle-school, ordinary
  conversation, longer dialogue, Python, C#, Unity, and general instruction
  following.
- Added a 30-chat evaluation covering casual conversation, memory, school
  math/science, reading, digital literacy, Python, C#, Unity, constraints,
  creativity, and tiny inputs.
- Added a 24-case Python/C# mastery evaluation.
- Added temperature-safe delays to pretraining and supervised fine-tuning.
- Added foundation-corpus tooling using Databricks Dolly 15k and TinyStories.
- Trained and compared foundation, broad-recovery, code, K–8, and focused
  prompt-binding candidates.

## Results

- Selected production checkpoint: 10/10 established challenge suite.
- Selected checkpoint on the harder raw 30-chat suite: 0/23 automatic checks,
  plus 7 manual-review chats. This exposes a real generalization limitation.
- Best experimental acceptance candidate: 3/23 automatic checks. It was not
  deployed because its other replies showed severe cross-topic contamination.
- Other evaluated broad candidates: 0/23 automatic checks.

These results are intentionally reported without inflating manual scores or
counting malformed answers as passes. Reaching a dependable 15/30 on held-out
raw prompts will require substantially more pretraining or a larger base model,
not another narrow memorization pass.

## Intentionally unchanged

No runtime fallback, math, identity, greeting, or LGBTQ response logic was
added, removed, or edited during this retraining pass. Music V1 and unrelated
Discord behavior were also left unchanged.

## Data notes

Databricks Dolly 15k is distributed under CC BY-SA 3.0. TinyStories is
distributed under CDLA-Sharing-1.0. Generated foundation artifacts and raw
downloads remain excluded from Git; their builders and configurations are
versioned for reproducibility.
