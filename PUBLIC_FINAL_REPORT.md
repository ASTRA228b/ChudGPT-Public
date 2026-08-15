# ChudGPT-Public final quality report

## Final verdict

The deployed `public_v10_balanced` checkpoint remains the strongest validated
Public checkpoint. It is still a small 20,999,184-parameter experimental model
and is not generally reliable. Public is intended to be the strongest general
ChudGPT release, but the available evidence does not justify claiming that it
is better at every prompt or comparable to a frontier assistant.

The new v18 candidate was **not deployed**. Its lower validation loss did not
translate into useful replies: it answered `What is 7 + 5?` with climate text,
returned Python for a Unity idea, and failed basic identity questions.

## What changed

- Added four-candidate neural sampling with relevance and fluency ranking.
- Penalized unrequested code, arithmetic, identity, and unrelated-topic leaks.
- Kept recent conversation turns while limiting self-contamination from old,
  malformed model replies.
- Added narrow, auditable repair for ChudGPT identity/family metadata.
- Added honest handling for invented ChudGPT profile names; the server never
  promotes a user-invented suffix into the list of real profiles.
- Added a reviewed glossary for explicitly named memes. Ordinary conversation
  still comes from the neural model; the glossary is not a general fallback.
- Added safe Windows/OneDrive checkpoint staging and weight-only initialization
  for clean foundation-to-SFT experiments.

## Data and training investigation

The v18 foundation corpus used 36,449 unique reviewed English OpenAssistant
documents after rejecting 47,870 deleted, unreviewed, malformed, duplicate, or
out-of-range rows. It contained 4,954,015 training tokens and 267,403 validation
tokens. The response-only SFT set contained 8,493 unique conversations: 8,423
broad conversations and 70 capped project-specific identity/meme examples.

Foundation validation improved from 4.0778 at step 200 to 3.8130 at step 1000.
SFT validation reached 3.4500 at step 100 and improved again by step 200. The
candidate nevertheless failed behavioral evaluation, proving that validation
loss alone was not a safe deployment criterion.

## Tests and benchmarks

- Focused runtime/data tests: **15 passed**.
- Current final production broad benchmark: **10/37** in the recorded run.
- Prior best stochastic v10 selected-candidate run: **12/37**.
- Rejected v18 SFT step 200: failed greetings, identity, arithmetic, knowledge,
  and code examples despite improved validation loss.
- Exact held-out-prompt checks found no representative benchmark prompts in the
  v18 foundation or SFT training sources.

The production benchmark varies because generation is sampled. The important
result is that v18 was clearly worse and therefore stayed offline.

## Remaining weaknesses

Public can still produce malformed grammar, irrelevant answers, bad arithmetic,
and broken code when every neural candidate is poor. Candidate ranking can pick
the least-bad generated answer but cannot manufacture knowledge missing from the
weights. Eliminating these failures requires a substantially larger, cleaner
pretraining corpus and more compute, or a larger architecture—not more repeated
templates or a hidden generic fallback.
