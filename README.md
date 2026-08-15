# ChudGPT-Public

## ChudGPT Desktop

The native ChudGPT Desktop client now lives entirely in [`desktop/`](desktop/README.md). It provides persistent multi-chat history, local search and data controls, a futuristic native interface, safe ChudGPT-Public API integration, and Windows/macOS/Linux release packaging without bundling the model.

ChudGPT-Public is an independently trained, experimental conversational language model and public web API. It has **20,999,184 trainable parameters** and is designed for general conversation, basic facts, arithmetic, and simple Python, C#, JavaScript, and Unity questions.

### Public V20

V20 is the current strongest Public serving profile. It combines the independently trained 20,999,184-parameter checkpoint with a local reviewed-response layer, exact decimal and large-integer arithmetic, conservative slang normalization, protected identity/meme facts, multi-candidate neural selection, and a dedicated Discord context mode. It does not call Pro, ChatGPT, or any external model.

The V20 corpus contains 8,914 unique cleaned conversations. Its final focused CUDA tune used 2,241 unique conversations: all 641 Public-authored conversations plus 1,600 reviewed conversations. A first broad V20 attempt was archived after it scored worse; the final candidate was selected only after held-out and stateful Discord tests.

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

That runs data preparation, base pretraining, response-only fine-tuning, and evaluation. Timestamped logs are stored in `reports`. The Public API checkpoint is selected in `serving_config.json`; V20 currently uses `checkpoints/public_v20_final/best.pt`. V8 through V18 and the rejected broad V20 candidate remain archived and selectable.

The official Discord bot sends a protected Discord-only system instruction plus scoped server, channel, and speaker metadata. Set `CHUDGPT_DEVELOPER_USER_ID` to Astra's numeric Discord user ID to identify the real developer account securely. Its conversations are logged as monthly JSONL files under `D:\ChudGPT-Discord-Logs` by default; set `CHUDGPT_DISCORD_LOG_DIR` to change that location. Tell server members if their bot messages are being logged.

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
