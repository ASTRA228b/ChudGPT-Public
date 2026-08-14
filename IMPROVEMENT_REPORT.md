# ChudGPT-Public v8 improvement report — 2026-08-14

## Final result

Public now serves `checkpoints/public_v8/best.pt`, family-aware alignment step 150, on CUDA. The architecture is unchanged at exactly **20,999,184 parameters**, an 8,192-token vocabulary, and a 1,024-token model context.

- Full held-out benchmark: **168/305** (previous Public: 161/305; unchanged Pro baseline: 106/305)
- Focused quality-per-token suite: **99.41/100** (before: 54.35)
- Python tests: **13/13 passed**
- Desktop tests: **5/5 passed**
- Desktop TypeScript, Vite, and Electron production builds: passed

## Math false-positive root cause and fix

Math intent existed in three inconsistent forms: serving, retrieval, and response scoring. Broad numeric/token overlap could label years, a lone number, `Nothing`, or `No math` as mathematical. Retrieval also discarded negation and accepted weak one-word matches.

Public now uses one conservative classifier. Math activates only for explicit expressions plus positive calculation wording, word operations, percentages, calculation-bearing conversions, or well-formed word problems. Negative intent wins first. The strict retriever requires the same intent and meaningful lexical evidence, and it abstains for corrections and ambiguous short turns.

Focused results:

- Math false positives: **90.00 → 97.31**
- True math: **50.00 → 100.00**
- Negation/corrections: **66.67 → 100.00**
- Short follow-up context: **20.00 → 100.00**

## Smarter per token

The old 30,000-row builder devoted roughly half of its raw capacity to arithmetic, price, and distance templates. Exact runtime math already solves those tasks, so that repetition taught little language behavior.

- Replaced the 30,000-row corpus with **21,900 unique, denser conversations**.
- Removed **8,100 low-value rows** from the selected corpus (27% smaller).
- Reduced raw arithmetic/price/distance template generation by **14,300 candidates**.
- Retained a balanced **6,000-conversation alignment set**.
- Final main corpus has **23,061 assistant messages** and maximum exact answer frequency **48**.
- Validation loss improved during each accepted stage: v5 1.9569 → 1.8566, v6 2.1454 → 2.0582, v7 2.2334 → 2.1939, v8 2.0416 → 2.0239.

## Meme and slang expansion

Public now has **108 audited meme/slang topics and formats**, up by 41 named topics. Coverage spans early image macros/rage comics, Doge, Pepe, Trollface, Wojak, Virgin vs Chad, gaming/Discord/programming culture, reaction formats, short-video conventions, irony/post-irony, and authored 2025–2026 terminology. Contextual examples teach usage and literal-versus-slang distinctions.

- Focused meme benchmark: **50.00 → 100.00**
- Full held-out meme category: **20/25**

## AI, self, “chud,” and model-family knowledge

Public can now explain:

- what AI is and why not every AI is a chatbot or conscious;
- that it is a small decoder-only transformer predicting tokens;
- exact parameter/context information, local helpers, memory, internet, consciousness, and reliability limits;
- that C.H.U.D. commonly expands to “Cannibalistic Humanoid Underground Dwellers” from the 1984 film;
- that online “chud” can be a disparaging label, while ChudGPT uses it as a playful project brand and not an insult toward the user;
- Buggy, Ultimate, Plus, Pro, Code, Mega, Public, and archived checkpoints 700/1300/1500/1600, including a clearly labeled opinion of their intended uses.

These descriptions use audited local project metadata. Public does not call Pro, sibling models, or an external AI.

## Retrieval changes

Retrieval now abstains on weak, short, and negated inputs; requires matching intent; rejects cross-topic nearest neighbors; and no longer gets an unconditional score bonus. Once a same-intent match clears the strict threshold, its audited answer can compete with neural candidates. This preserved factual/coding performance while preventing `Nothing` or `No math` from retrieving unrelated examples.

## Full benchmark and regressions

Public v8 scored **168/305**, seven points above the prior 161/305 result and 62 above the unchanged Pro baseline of 106/305. Perfect categories were arithmetic (25/25), word problems (25/25), memory (20/20), and adversarial routing (20/20).

Remaining weaknesses are real: knowledge 4/25, common sense 6/25, strict instructions 6/25, coding 7/25, debugging 6/25, references 4/20, and the legacy identity suite 12/20. Four neural candidates plus retrieval also increase latency. This is a materially better small experimental model, not a ChatGPT-class system.
