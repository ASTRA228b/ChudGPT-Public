# ChudGPT-Public v11 experiment report

## Outcome

The v11 checkpoint was **rejected and not deployed**.  It reduced validation
loss, but it did not improve unseen conversational behavior.  The existing v10
Public checkpoint remains the served model.

## Root causes found

- The original project-authored corpus is dominated by synthetic templates,
  especially arithmetic and short code transforms.
- The old pretraining serializer repeated the full permanent system prompt in
  every document.  In the first v11 preparation, roughly 2.5 million of 3.1
  million tokens were that repeated prompt rather than useful language.
- A random corpus sampler preserved severe category imbalance.  An audit found
  11,730 math conversations versus only 508 identity and 224 meme conversations
  before repetition filtering.
- A 21M-parameter model trained on this small templated corpus learns local
  phrases and low loss but does not reliably learn prompt intent.  Candidate
  reranking cannot repair a run where all candidates are malformed.

## Changes made

- Added a compact pretraining-only system identity while preserving the full,
  protected Public system prompt for chat and response-only SFT.
- Added `build_public_v11_corpus.py`, which filters known malformed/template
  leakage, deduplicates conversations, caps repeated assistant outputs, and
  enforces explicit category quotas.
- Rebuilt a clean 4,096-token byte-level BPE tokenizer.
- Added isolated 21,004,308-parameter v11 model, training, and SFT configs.
- Added a 37-turn broad evaluation covering identity, family knowledge, AI,
  casual/short prompts, memes, math, code, Unity, knowledge, common sense,
  uncertainty, nonsense, memory, pronouns, topic changes, and follow-ups.
- Strengthened neural candidate ranking against unrequested code, accidental
  arithmetic, greeting-topic leakage, repetition, and unrelated domains.  It
  still ranks model generations; it does not insert a generic fallback answer.
- Added four regression assertions for cross-topic leakage and requested output
  types.  The full unit suite passes 12/12.

## Dataset and tokenizer

- 7,900 unique conversations
- 5,000 general conversation/knowledge
- 1,400 math
- 850 code
- 500 identity
- 150 meme/slang
- Maximum exact assistant-response repetition: 48 (identity variants)
- Processed tokens: 371,739 train; 19,628 validation
- Broad evaluation contamination: 26/28 single-turn prompts are exact-unseen;
  the intentionally minimal prompts `what` and `67` overlap.

## Training results

Base pretraining reached step 150:

- step 25 validation loss 3.9756 / perplexity 53.28
- step 50 validation loss 2.5803 / perplexity 13.20
- step 100 validation loss 1.6947 / perplexity 5.45
- step 150 validation loss 1.4970 / perplexity 4.47

Response-only SFT reached step 400:

- final validation loss: 2.1883

## Behavioral results

The current v10 production checkpoint scored 5/37 on the new broad suite.  The
v11 base and v11 SFT checkpoints both failed the fixed unseen qualitative suite,
so they were rejected before deployment.  Representative v11 failures included:

- `What is 7 + 5?` -> `10% of range.`
- `Explain why the sky is blue.` -> `The main reason is that water can keep travel casual.`
- `Give me an idea for a Unity game.` -> the same unrelated water/travel phrase
- Python requests sometimes emitted incomplete or mixed-language code.

This is evidence that loss improved without useful generalization.  Claiming
v11 was smarter would be misleading.

## Remaining limitation and next credible path

Public needs a substantially larger, genuinely varied natural-language corpus,
not more permutations of a few response templates.  The next credible model
experiment should use millions of clean, diverse tokens with semantic-family
held-out validation and should only deploy after raw neural outputs outperform
v10.  Runtime reranking can block some category leakage, but it cannot turn four
bad candidates into a good answer without becoming the fallback system the
project intentionally avoids.
