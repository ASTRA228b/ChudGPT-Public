# Landing-page update prompt

Update the ChudGPT landing page to present **ChudGPT-Public V20** as the strongest and recommended general-purpose ChudGPT release.

Use accurate wording: V20 is a custom 20,999,184-parameter decoder-only transformer with a 1,024-token model context. It uses a newly focused CUDA-tuned checkpoint plus exact math, reviewed local knowledge/code responses, slang normalization, multi-candidate relevance checks, and an improved Discord mode. It does not call ChudGPT Pro, ChatGPT, or an external model.

Mention that the V20 clean corpus has 8,914 unique conversations and its focused final curriculum used 2,241 unique conversations. Add highlights for large-number and decimal math, Python/C#/JavaScript/Unity help, Discord server/user context, Astra developer recognition by Discord ID, safer mentions, conversation clearing, and improved topic switching. Link the Public website/API and desktop download from the existing Public card.

Keep the limitations visible: it is a small local experimental model, has no live internet, can still make mistakes on novel questions, and is not suitable for high-stakes advice. Do not claim frontier-model performance.

Add a short **Discord reliability update** note: the official bot now understands its server, channel, current speaker, real Discord user mention, member roles, and Astra/developer relationship. It explicitly mentions the requesting user, preserves safe member-directed mentions, explains `!chud <message>` and `!chud clear`, recognizes common Discord shorthand including `smt`, understands GTAG/Gorilla Tag, handles personal identity questions respectfully without guessing, protects private host paths, and rejects online-game cheats while offering legitimate Unity examples for private projects. State that 66 Public tests and 5 Discord-bot tests pass. Do not expose or link private Discord conversation logs.

Update the bot-test number to **6/6** and mention that the official Discord bot automatically falls back to its same-host CUDA API when the public Vercel/Cloudflare route is temporarily unavailable, keeping Discord chat operational during tunnel outages.

Update the Discord bot-test number again to **7/7**. Add that first-person and third-person developer identity are now distinct: the bot identifies the current speaker from Discord metadata and answers who Astra/the developer is using verified application-owner metadata instead of neural guessing.
