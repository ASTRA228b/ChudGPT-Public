# ChudGPT-Public improvement report — 2026-08-13

## Result

Public now uses the 20,999,184-parameter checkpoint at `checkpoints/public_v4/best.pt`, step 800 of its final response-alignment stage, on CUDA.

The identical live 305-case benchmark scored:

- ChudGPT-Public: **161/305**
- ChudGPT Pro (unchanged): **106/305**

On the original 280 cases, excluding the subsequently added meme category, Public improved from **73/280 before** to **138/280 after**.

## Root cause

The previous 20,000-row dataset contained only 4,228 unique conversations. Two generic final responses appeared 2,000 times each, 600 assistant messages contained mojibake, and narrow templates dominated. The API then returned the first non-empty generation without using its quality checker. A greeting could therefore select arithmetic such as `The answer is 462.`

## Changes

- Rebuilt the main corpus as 30,000 unique project-authored conversations.
- Reduced maximum exact assistant-answer frequency from 2,000 to 44.
- Removed detected mojibake from the rebuilt corpus.
- Added a balanced 6,000-conversation alignment set.
- Added representative meme literacy from 2016–2026.
- Added decimal arithmetic and general constant-speed word-problem helpers.
- Added four neural candidates, response-type scoring, training-data-leak rejection, and repetition/topic checks.
- Added retrieval-guided generation using only Public's cleaned local examples; no Pro routing or external model API is used.
- Added randomized language selection for underspecified `Code me some code` requests.
- Added 305 held-out API tests across 13 categories.

## Previously failing prompts

- `Hello! What can you do?` now returns a relevant capability description.
- The 65 mph for 2.5 hours problem returns 162.5 miles.
- Steel versus feathers explains that both have equal one-kilogram mass.
- `What are you?` returns the ChudGPT Public identity.
- The 6-7 meme receives its meme context rather than arithmetic.
- Generic code requests return a complete code block in a randomly selected supported language.

## Remaining weaknesses

This is still a very small model. Broad general knowledge scored 3/25, common sense 6/25, instruction following 6/25, coding 7/25, and memory 4/20 on this strict automatic benchmark. Retrieval can still pick a nearby rather than exact topic. Four-candidate inference also increases latency. These limitations are reported rather than hidden.
