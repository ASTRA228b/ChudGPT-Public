# Landing-page update prompt: ChudGPT-Public relevance update

Update only the ChudGPT-Public card and Public details on the ChudGPT landing page. Preserve every other model card and route.

Use these verified details:

- Name: ChudGPT-Public
- Badge: PUBLIC · 21M
- Exact parameters: 20,999,184
- Current checkpoint: Public v4 alignment step 800
- Dataset: 30,000 unique cleaned conversations plus 6,000 balanced alignment conversations
- Serving: CUDA, four candidate generations, relevance/response-type ranking, and retrieval-guided generation from Public's own cleaned local examples
- New coverage: everyday conversation, arithmetic and word problems, simple code/debugging, recent-session references, identity, and representative meme literacy from 2016–2026
- Measured benchmark: Public 161/305 versus unchanged Pro 106/305 on identical prompts; original non-meme Public score improved from 73/280 to 138/280
- Limitations: still a small experimental model; broad facts, common sense, strict instruction following, coding, and memory remain imperfect; no live internet
- Primary link: https://chudgpt-public.vercel.app/
- API status: https://chudgpt-public.vercel.app/api/status

Add a concise note such as “Relevance update: larger unique corpus, balanced alignment, 305-case evaluation, and 2016–2026 meme literacy.” Do not claim frontier-model quality or perfect accuracy. Keep the card responsive on phones.
