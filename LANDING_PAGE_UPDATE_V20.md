# Landing-page update prompt

Update only the ChudGPT-Public V20 card/release notes on the ChudGPT landing page.

Describe this as a major log-driven quality and reliability pass. The audit fully reviewed 691 valid Discord exchanges (516 unique prompts) without publishing private log contents. Public now uses a rebalanced 6,862-conversation clean corpus and a 1,669-conversation focused curriculum: 669 authored Public conversations, 700 reviewed normal-prose conversations, 120 legitimate structured requests, and 180 code tasks. State that 2,080 topic/template-leaking examples and 151 malformed or low-quality examples were removed.

Highlight stronger relevance checking, no neural-candidate bypass after rejection, better short follow-up recovery using the preceding exchange, fewer unsolicited tutorials, stronger repetition/recursive-loop rejection, explicit programming-language consistency, exact response constraints, improved Discord speaker/context continuity, and intent-preserving multilingual conversation for English, Spanish, Portuguese, French, German, Russian, and Japanese.

Mention that a new CUDA quality checkpoint was fine-tuned for 250 optimizer steps from the previous V20 checkpoint. Validation loss reached 3.6326, while the prior V20 checkpoint remains archived and switchable. Verified results are 210/210 Public tests, 32/32 Discord-bot tests, and 18/18 standard live serving benchmark cases.

Keep all technical claims accurate: ChudGPT-Public V20 is a custom 20,999,184-parameter decoder-only transformer with a 1,024-token model context. It uses exact local math, reviewed local reliability paths, candidate rejection/resampling, and an optional Google translation layer in the Discord bot. It does not call Pro, ChatGPT, or an external language model.

Keep limitations visible: Public is a small experimental model, has no live internet, can still produce mistakes or nonsense on novel prompts, and is not appropriate for high-stakes advice. Do not claim frontier-model quality or expose private Discord logs, user IDs, tokens, server details, or local file paths.
