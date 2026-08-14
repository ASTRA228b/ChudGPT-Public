# ChudGPT-Public nonsense investigation (v12-v14)

## Verdict

No experimental checkpoint was deployed. Public remains on the existing v10
checkpoint because neither corrective run produced a reliable neural model.

## Root cause

The problem is in the learned weights, not the Vercel page or JSON API. The old
corpus stored many multi-turn records whose individual turns were dominated by
arithmetic, tiny code transforms, repeated project metadata, and canned topic
starters. Conversation-level quotas hid that turn-level imbalance. The model
therefore learned locally plausible fragments without reliable intent or topic
tracking. When every sampled candidate is malformed or irrelevant, candidate
ranking cannot create a correct answer.

## Data cleanup

`build_public_v13_corpus.py` now flattens conversations into individual
user/assistant training pairs before balancing and deduplication. It uses only
project-owned data and rejects malformed Unicode, training references, demo
identifiers, known canned phrases, secret-number/puzzle templates, and math
boilerplate.

The resulting corpus contains 5,416 unique turn pairs:

- 2,584 general conversation/knowledge
- 1,000 math
- 1,092 code
- 593 identity
- 128 meme/slang
- 19 uncertainty

The audit found 3,355 numeric-only assistant turns in the prior supposedly
balanced corpus. The new builder classifies and limits these at the turn level.

## Experiments

- v12 tested compact prompt alignment on the prior v11 lineage. It still mixed
  identity, facts, arithmetic, and code, so it was rejected.
- v13 trained 500 response-only steps on the clean turn-balanced corpus. Final
  broad raw score: **7/37**. It was rejected.
- v14 applied the same clean corrective corpus to the stronger deployed v10
  lineage for 400 steps. Validation loss improved from 2.9293 at step 50 to
  2.4235 at step 400, but broad raw behavior scored **6/37**. It was rejected.

Examples from rejected v14 include Saturn being answered with mixed identity
text, a Unity request producing broken JavaScript, and seventeen plus
twenty-six producing malformed prose. These results prove that lower validation
loss on the present corpus does not equal useful generalization.

## Runtime safeguards

Candidate ranking now adds penalties for unrequested model-identity leakage,
replacement characters, glued/malformed tokens, unbalanced code fences, very
long invented words, and replies lacking ordinary connective language. It
still selects only model-generated candidates and does not provide a canned
general fallback.

## Credible next training path

The 21M architecture needs millions of genuinely varied clean natural-language
tokens before conversational SFT. Repeating or templating a few thousand custom
answers will deepen the contamination. A future base run should use semantic
held-out evaluation, train substantially longer, and be deployed only when raw
neural answers beat v10 across unseen conversation, facts, math, and code.

