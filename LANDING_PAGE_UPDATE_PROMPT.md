# Landing-page update log — ChudGPT-Public v9 raw-model release

Update only the ChudGPT-Public card, detail section, and update log. Preserve every other model and existing desktop release entry.

Add an update titled **Public v9: raw model, zero conversational fallbacks**.

- Exact model size: **20,999,184 parameters**; vocabulary: 8,192; context: 1,024 tokens.
- Selected checkpoint: `public_v9_refined/best.pt`, CUDA step **600**, after an earlier 300-step v9 candidate.
- The API now returns raw checkpoint generation. Arithmetic, greetings, identity, jokes, code, retrieval, comparisons, corrections, fact-memory, quality selection, and keyword-answer substitution were removed.
- Empty decoding is retried only as technical recovery; repeated empty decoding returns an API error instead of conversational content.
- Dataset: **9,000 unique clean conversations**, split into 8,280 training and 720 validation records. Malformed Unicode and known leaked templates were removed; exact answer repetition is capped at four; held-out prompts are excluded.
- Response-only validation loss: **1.5104**.
- Raw held-out short-context score: **14/40 → 19/40**.
- Existing raw eight-case benchmark: **0/8 → 0/8**.
- Tests: **8/8 passed**.
- Production reports CUDA, step 600, `raw_model_generation: true`, `conversational_fallbacks: false`, and `response_substitution: false`.

Suggested card copy:

> **Public v9 — raw 21M experiment.** No canned conversational fallbacks or hidden answer substitution. Short-context training improved the held-out raw score from 14/40 to 19/40, but the model still produces frequent nonsense and should not be trusted for important facts, math, or code.

Do not claim that v9 is accurate, reliable, generally intelligent, or comparable to a frontier model. Keep the existing desktop download information and mobile responsiveness.
