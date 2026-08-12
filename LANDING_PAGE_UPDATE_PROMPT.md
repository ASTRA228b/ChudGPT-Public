# ChudGPT-Public landing-page update prompt

Update the ChudGPT landing page to add a new model card for **ChudGPT-Public** without changing or removing any existing model cards.

Use these details:

- Name: ChudGPT-Public
- Badge: PUBLIC · 21M
- Exact parameters: 20,999,184
- Description: An independently trained, experimental general-purpose ChudGPT model with conversational answers, general information, basic math, and simple Python, C#, JavaScript, and Unity coding support.
- Availability: Public website and HTTPS API hosted through Vercel, with inference running on the project owner's CUDA server.
- Limitations: Small experimental model; may be inaccurate; no live internet; availability depends on the owner-hosted inference server.
- Primary button: **Try ChudGPT-Public**
- Primary link: replace `CHUDGPT_PUBLIC_VERCEL_URL` with the deployed Vercel URL.
- Secondary link: **API status**, pointing to `CHUDGPT_PUBLIC_VERCEL_URL/api/status`.

Place it in the model/checkpoint library near ChudGPT Plus, Pro, Code, and Mega. Keep the existing design language, make the card responsive on phones, and do not claim it performs like a frontier model. Mention that developers can call `POST /api/chat` with JSON containing `message` and an optional `session_id`.
