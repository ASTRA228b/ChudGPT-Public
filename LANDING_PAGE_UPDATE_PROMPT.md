# Full update-log prompt for the ChudGPT-Pro / landing-page chat

Identity update: ChudGPT Public now answers natural self-identity questions consistently, including short wording such as "What are you?" and follow-ups such as "What are you fully?". Its answer explains that Public is a 20,999,184-parameter decoder-only transformer with an 8,192-token vocabulary and 1,024-token model context, plus local retrieval, arithmetic, session recall, and response-quality checks. It also clearly states its lack of consciousness, live browsing, and cross-chat memory.

Update the landing page’s **ChudGPT-Public** card and Public detail section using this verified release log. Do not modify the behavior or claims of Pro, Plus, Buggy, Ultimate, Code, Mega, or archived checkpoint cards. Preserve all routes and mobile responsiveness.

## ChudGPT-Public v8 — smarter-per-token update

- Exact model size: **20,999,184 parameters** (21M badge)
- Selected checkpoint: **Public v8 family-aware alignment, step 150**
- Runtime: CUDA, 1,024-token model context, four neural candidates, strict local retrieval, exact math, and response-quality selection
- Data: **21,900 unique dense conversations + 6,000 balanced alignment conversations**
- Cleanup: removed **8,100 low-value selected rows** and reduced raw repetitive math/price/distance templates by **14,300 candidates**; corpus is 27% smaller but more informative per token
- Meme literacy: **108 audited topics/formats**, spanning early internet image macros and rage comics through authored 2025–2026 slang and formats
- New self-knowledge: explains AI, its decoder-only transformer design, exact limits, memory/internet/consciousness, what “chud” means, and why ChudGPT uses the name
- Model-family knowledge: can describe and give a clearly labeled opinion about Buggy, Ultimate, Plus, Pro, Code, Mega, and checkpoints 700/1300/1500/1600 without calling those models
- Math routing: one conservative classifier now prevents numbers, years, `Nothing`, and `No math` from accidentally triggering arithmetic
- Retrieval: abstains on weak/short/negated matches and requires same-intent evidence
- Focused quality-per-token score: **54.35 → 99.41/100**
- Focused meme score: **50 → 100**
- Math false-positive score: **90 → 97.31** while true-math score improved **50 → 100**
- Full held-out benchmark: **Public 168/305**, previous Public 161/305, unchanged Pro baseline 106/305
- Verified: 13/13 Python tests, 5/5 desktop tests, TypeScript checks, Vite build, and Electron build passed

Suggested card line:

> Public v8: stricter intent detection, 108-topic meme literacy, stronger short-context handling, complete AI/self/model-family knowledge, and denser training—168/305 held-out.

Keep the honest limitation text: Public is still a small experimental local model with weak broad knowledge, common sense, strict instruction following, deep reasoning, and coding. It has no live internet and can confidently make mistakes.

Primary link: https://chudgpt-public.vercel.app/

API status: https://chudgpt-public.vercel.app/api/status
