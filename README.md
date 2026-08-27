# ChudGPT-Public

## ChudGPT Desktop

The native ChudGPT Desktop client now lives entirely in [`desktop/`](desktop/README.md). It provides persistent multi-chat history, local search and data controls, a futuristic native interface, safe ChudGPT-Public API integration, and Windows/macOS/Linux release packaging without bundling the model.

ChudGPT-Public is an independently trained, experimental conversational language model and public web API. It has **20,999,184 trainable parameters** and is designed for general conversation, basic facts, arithmetic, and simple Python, C#, JavaScript, and Unity questions.

### ChudGPT-Public-Music V1

Music V1 is a separate 20,999,184-parameter checkpoint fine-tuned for original lyrics, hooks, titles, song concepts, and musical style ideas. It has isolated conversation sessions and does not replace the standard Public V20 checkpoint. Its personality deliberately permits funny, chaotic, and occasionally nonsensical writing. It remains a very small experimental model: rhyme, factual music knowledge, and instruction following are not dependable.

The expanded Music corpus contains 2,798 unique conversations, including complete songs, vague real-world requests, genre/title selection, revisions, and multi-turn title/style recall. A two-stage CUDA response-only update ran for 500 then 400 steps; held-out validation loss reached 0.7541 after the first stage and 0.6448 after the continuation. The checkpoint is generated locally at `checkpoints/public_music_v1/best.pt` and is excluded from Git because of its size.

Music serving is deliberately pure generation. It has no canned lyric engine, keyword answer table, reviewed-response layer, or fallback song text. The runtime samples multiple complete neural drafts and ranks those generated drafts for requested structure and conversation continuity. Because this remains a 21M-parameter experiment, outputs can still be malformed, repetitive, or forget details.

Build and train it independently:

```cmd
python build_music_v1_data.py
python fine_tune.py --config configs/finetune_music_v1.yaml --device cuda
```

Run both Public and Music checkpoints:

```cmd
python public_api_server.py --device cuda --port 8010 --music-checkpoint checkpoints/public_music_v1/best.pt
```

Music endpoints are `GET /api/music/status`, `POST /api/music/chat`, and `POST /api/music/clear`. Every Music status/chat response includes `"music": true`. The website is available at `/music`, Discord uses `!chud music <prompt>`, and ChudGPT Desktop 0.3.1 adds a Public/Music model selector.

### Public V20

V20 is the current strongest Public serving profile. It combines the independently trained 20,999,184-parameter checkpoint with a local reviewed-response layer, exact decimal and large-integer arithmetic, conservative slang normalization, protected identity/meme facts, multi-candidate neural selection, and a dedicated Discord context mode. It does not call Pro, ChatGPT, or any external model.

The current V20 corpus contains 6,912 unique cleaned conversations after removing 2,080 topic/template leaks and 151 malformed or low-quality rows. Its balanced quality tune used 1,719 conversations: 719 Public-authored conversations, 700 reviewed prose conversations, 120 legitimate structured requests, and 180 reviewed code tasks. Earlier V20 checkpoints remain archived and selectable.

### Expanded emoji awareness

Public V20 uses the complete cached metadata shipped by `emoji` 2.15.0: 5,225 Unicode sequences through Emoji 17.0. It recognizes emoji-only messages, multi-codepoint ZWJ sequences, skin-tone variants, flags, common colon aliases, classic emoticons, and Discord static/animated custom emoji names. A compact model-only annotation supplies possible meanings while preserving the original message; surrounding conversation still decides whether `😭`, `💀`, or `🔥` is literal, serious, sarcastic, celebratory, or meme-like. Reactions to recent bot messages are remembered as context for the next turn but do not trigger automatic replies.

This is semantic assistance, not a claim that a 21M-parameter checkpoint perfectly understands every emoji or culture-specific use. Unknown or ambiguous usage remains generative, and context can still be misread.

It is a small custom model—not ChatGPT and not a frontier model. It can be inaccurate, has no live internet access, and should not be trusted for medical, legal, financial, or safety-critical decisions.

## Architecture

| Setting | Value |
|---|---:|
| Parameters | 20,999,184 |
| Vocabulary | 8,192 BPE tokens |
| Context length | 1,024 tokens |
| Embedding width | 384 |
| Transformer layers | 9 |
| Attention heads | 6 |
| Feed-forward width | 1,808 |
| Positional encoding | RoPE |

The model is a decoder-only causal transformer implemented in Python and PyTorch. Public uses response-only fine-tuning, multi-candidate generation, response-type/relevance scoring, conservative intent detection, and strict local use of reviewed examples.

Public now has explicit self-knowledge: it can explain artificial intelligence, its own 20,999,184-parameter transformer architecture and limitations, the film/slang/project meanings of “chud,” and the roles of Buggy, Ultimate, Plus, Pro, Code, Mega, and the archived numbered checkpoints. Those descriptions come from audited project metadata; Public does not call the sibling models to answer.

## Project layout

```text
ChudGPT-Public/
├── web/static/                  complete Vercel site and serverless proxy
├── chudlm/                      model, data, prompting, and generation code
├── configs/                     model and training settings
├── data/public_conversations.jsonl
├── tests/
├── prepare.py
├── train_public.py
├── finetune_public.py
├── evaluate_public.py
├── public_api_server.py         local CUDA inference API
├── start_training.cmd
├── train_pipeline.ps1
└── VERCEL_DEPLOYMENT.md
```

## Requirements

- Windows 10/11 or another Python-compatible operating system
- Python 3.10+
- 16 GB system RAM recommended
- NVIDIA GPU with 8 GB VRAM recommended for training
- CPU inference works but is slower
- Node/Vercel is only used for the lightweight website/API proxy

Install dependencies:

```cmd
cd /d C:\path\to\ChudGPT-Public
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For an NVIDIA GPU, install a CUDA-enabled PyTorch build appropriate for the machine if the default package does not provide CUDA.

## Prepare and train

The easiest complete run on this machine is:

```cmd
cd /d C:\Users\admin\OneDrive\Documents\ChudGPT\ChudGPT-Public
start_training.cmd
```

That runs data preparation, base pretraining, response-only fine-tuning, and evaluation. Timestamped logs are stored in `reports`. The Public API checkpoint is selected in `serving_config.json`; V20 currently uses `checkpoints/public_v20_quality/best.pt`. The previous V20 checkpoint, V8 through V18, and the rejected broad V20 candidate remain archived and selectable.

The official Discord bot sends a protected Discord-only system instruction plus scoped server, channel, and speaker metadata. It securely treats the Discord application owner as Astra/the developer; `CHUDGPT_DEVELOPER_USER_ID` can explicitly override that account. Its conversations are logged as monthly JSONL files under `D:\ChudGPT-Discord-Logs` by default; set `CHUDGPT_DISCORD_LOG_DIR` to change that location. Tell server members if their bot messages are being logged.

To switch versions, stop the Public server, edit only `selected_checkpoint` in `serving_config.json`, and restart it. You can also perform a one-run override without editing the file:

```bat
python public_api_server.py --checkpoint checkpoints/public_v10_balanced/best.pt --device cuda
```

To reproduce the two Public-only improvement stages after the original chat checkpoint:

```cmd
python build_public_data.py
python fine_tune.py --config configs/public_v3.yaml --device cuda
python build_alignment_data.py
python fine_tune.py --config configs/public_v4_alignment.yaml --device cuda
python fine_tune.py --config configs/public_v5_dense.yaml --device cuda
python fine_tune.py --config configs/public_v6_alignment.yaml --device cuda
python fine_tune.py --config configs/public_v7_identity.yaml --device cuda
python fine_tune.py --config configs/public_v8_family.yaml --device cuda
```

Individual commands:

```cmd
python prepare.py
python count_parameters.py
python train_public.py --device cuda
python finetune_public.py --device cuda
python evaluate_public.py --device cuda
python chat_public.py --device cuda
```

Training checkpoints are intentionally excluded from Git because they are large generated artifacts. Do not claim an untrained or one-step smoke checkpoint is a completed model.

## Run the CUDA API

After training finishes:

```cmd
cd /d C:\Users\brian\OneDrive\Documents\ChudGPT\ChudGPT-Public
C:\tmp\ChudGPT-venv\Scripts\python.exe public_api_server.py --device cuda --port 8010
```

Test it locally:

```cmd
curl http://127.0.0.1:8010/api/status
curl -X POST http://127.0.0.1:8010/api/chat -H "Content-Type: application/json" -d "{\"message\":\"What is 7 + 5?\",\"session_id\":\"demo\"}"
```

Endpoints:

- `GET /api/status`
- `POST /api/chat` with `message` and optional `session_id`
- `POST /api/clear` with `session_id`
- `GET /api/music/status`
- `POST /api/music/chat` with `message` and optional `session_id`
- `POST /api/music/clear` with `session_id`

## Publish with Vercel and Cloudflare

Vercel does **not** run the CUDA/PyTorch model. It hosts the website and serverless HTTPS proxy. Your PC or mini-server runs `public_api_server.py`, and Cloudflare Tunnel connects it to Vercel.

1. Push this repository to GitHub.
2. Start the CUDA API on port 8010.
3. Start a Cloudflare tunnel:

   ```cmd
   C:\Users\brian\OneDrive\Documents\ChudGPT\tools\cloudflared.exe tunnel --url http://127.0.0.1:8010
   ```

4. Import the GitHub repository into Vercel.
5. Set **Root Directory** to `web/static`.
6. Select **Other** as the framework and leave build/install/output commands blank.
7. Add `CHUDGPT_BACKEND_URL` using the printed `https://...trycloudflare.com` origin, without a trailing slash.
8. Deploy. The site and keyless public API will share the same Vercel domain.

Full instructions and an API-call example are in [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md).

The latest prompt-alignment diagnosis, training details, and measured before/after results are in [IMPROVEMENT_REPORT.md](IMPROVEMENT_REPORT.md).

Quick Cloudflare tunnel URLs change after restart. A named Cloudflare Tunnel with your own hostname is recommended for a stable deployment.

For Windows CMD API testing, keep `curl.exe` on one line; `\` is a Linux/macOS line-continuation character and does not work in CMD.

## Test

```cmd
pytest -q
```

Tests verify the exact parameter count, dataset uniqueness and encoding, tokenizer/model compatibility, API contract, Vercel files, response-quality routing, and the 305-case held-out evaluation definition.

Run the identical live Public-vs-Pro benchmark after both APIs are running:

```cmd
python benchmark_vs_pro.py --models both
```

## License and credit

See [LICENSE](LICENSE). Created by [ASTRA228b](https://github.com/ASTRA228b).
