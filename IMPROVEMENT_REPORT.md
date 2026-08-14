# ChudGPT-Public v8 improvement report — 2026-08-14

## No-fallback and random-code release

Public now has **no canned uncertainty fallback**. If all normal candidates fail relevance checks, a fresh neural repair is shown; if that repair is bad, the bad answer remains visible by design. This makes the small model feel less scripted, but it measurably reduces broad reliability.

Root causes addressed:

- Generic code requests were previously rewritten into another neural prompt and could collapse to `Explain` or an unrelated training fragment.
- Joke requests had no enforced response type, allowing arithmetic/template leakage.
- The desktop placeholder-like text `Message ChudGPT...` entered identity generation instead of behaving like an empty conversational invitation.
- The repeated meme suffix could survive inside non-meme replies; the quality checker now rejects that response shape outside meme intent.

Runtime changes:

- `Code Me Some Code`, `give me some random code`, and close equivalents now select among **nine languages**: Python, C#, JavaScript, C++, Java, Rust, Go, Lua, and TypeScript.
- Each random-code answer identifies its language, chooses between two useful program families, randomizes values, and returns a complete fenced program. This creates at least 18 structural combinations before randomized values.
- Funny-joke requests now produce a short actual joke rather than entering unrestricted generation.
- `Message ChudGPT...` is handled as a conversational invitation.
- There is still no API response cache or cross-session answer reuse.

Data changes:

- The main corpus remains **21,900 unique conversations** and alignment remains **6,000 conversations**.
- Added broader C++, Java, Rust, Go, Lua, and TypeScript examples plus joke and placeholder-message alignment examples.

Verification:

- Automated tests: **20/20 passed**.
- Exact desktop regression prompts: **13/14 behavior targets passed**. Arbitrary nonsense remains intentionally unconstrained because fallback output is forbidden.
- Previous fallback-enabled benchmark: **182/305**.
- Current strict no-fallback benchmark: **146/305**.
- Preserved strengths: arithmetic **25/25**, word problems **25/25**, memory **20/20**, adversarial **20/20**.

The 36-sample random-code regression requires at least three languages and eight distinct programs per run; the implementation supports nine languages. This release honestly trades benchmark reliability for raw model behavior on unrecognized prompts.

## Short-prompt relevance update

### Root causes

- When every neural and retrieved candidate failed validation, serving still selected the least-bad rejected candidate. This directly exposed unrelated arithmetic, books, recipes, and other topic leakage.
- Public's topic-dialogue generator produced 990 copies of the `One useful way into ...` starter in the main corpus and 700 in alignment data.
- Alignment data allowed a single assistant answer to repeat up to 38 times, giving generic response shapes too much influence.
- Identity routing recognized `What are you?` but not the equally direct `What is ChudGPT?` form.
- Meme names made of numbers or very short words were discarded by the normal content-word relevance filter.
- Code requests could remain technically valid candidates even when the candidate contained no code.

### Changes

- Rejected candidates can no longer reach the user. Candidate selection is repeated after the neural repair pass, and an intent-shaped concise uncertainty response is used only when every candidate remains invalid.
- Product identity classification now runs before slang classification, and `ChudGPT` is treated as one product name rather than as the standalone word `chud`.
- Added general checks for missing requested code, unrequested math templates, unrelated topic starters, and short meme-name anchors.
- Added compact training conversations for slang, unknown text, meme overview, and Unity movement code.
- Rebuilt both runtime corpora. Main corpus remains **21,900 unique conversations**; alignment remains **6,000 conversations**.

### Dataset audit, before → after

| Dataset | Max identical answer | `One useful way into...` rows | Unique assistant answers |
|---|---:|---:|---:|
| Main Public corpus | 48 → 32 | 990 → 0 | 16,839 → 16,902 |
| Alignment corpus | 38 → 8 | 700 → 0 | 3,925 → 4,495 |

Alignment duplicate instances fell from **2,990 to 2,226** (764 removed). The main corpus gained varied multi-turn answers while preserving its exact 21,900-conversation size.

### Verification

- Automated tests: **16/16 passed** (previously 13 tests).
- Full unchanged held-out benchmark: **182/305**, up from **168/305**.
- Preserved category scores: arithmetic **25/25**, word problems **25/25**, memory **20/20**, adversarial **20/20**.
- Meme score: **20/25**.

Observed corrections:

- `What is ChudGPT?` previously returned the second half of an insult dialogue; it now gives Public's direct identity.
- `you're a chud` previously wandered into books or corrupted math; it now gives a short, relevant playful response.
- `Tung tung tung tung tung sahur` previously produced arithmetic contamination; it now identifies the absurdist meme chant.
- `write a Unity movement script` previously returned debugging advice; it now returns a complete C# `MonoBehaviour`.
- Unknown keyboard text now gets a concise clarification instead of an unrelated memorized fact.

Remaining limitations: the 21M checkpoint is still weak on brand-new world knowledge, complex code, reference tracking, and some ordinary phrasings. Runtime relevance improved substantially, but this update does not turn the small checkpoint into a frontier model.

## Identity wording hotfix

Public now recognizes plain and conversational identity questions such as `What are you?`, `What are you fully?`, `Who are you really?`, `Tell me about yourself`, and `What kind of model are you?`. These forms return one consistent description of ChudGPT Public's architecture, parameter count, context, runtime helpers, limitations, and relationship to the other ChudGPT profiles instead of falling through to an unreliable neural generation.

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
