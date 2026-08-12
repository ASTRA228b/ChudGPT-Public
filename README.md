# ChudGPT-Public

ChudGPT-Public is a separate 21M-class conversational transformer designed to be trained locally and published as a self-contained Hugging Face model. It does not modify or share checkpoints with Buggy, Ultimate, Plus, Pro, Code, or Mega.

Author: [ASTRA228b](https://github.com/ASTRA228b)

## Architecture

| Setting | Value |
|---|---:|
| Trainable parameters | **20,999,184** |
| Vocabulary | 8,192 byte-level BPE tokens |
| Context length | 1,024 tokens |
| Embedding width | 384 |
| Transformer layers | 9 |
| Attention heads | 6 |
| Head width | 64 |
| Feed-forward width | 1,808 |
| Position encoding | RoPE |

The tokenizer is the project's proven natural-language tokenizer. A forced tokenizer trained only on the templated custom corpus was rejected because it produced just 2,559 useful tokens; pretending it had 12,288 would waste parameters.

## Dataset

`build_public_data.py` deterministically creates **20,000 project-authored conversations and 60,000 messages**. It covers model self-knowledge, general information, geography, science, computing concepts, arithmetic, stories, jokes, emotional support, Python, JavaScript, C#, and Unity. It does not download an external dataset.

Prepared corpus size on the current build:

- Prepared token totals are printed by `prepare.py` and recorded in `data/processed/metadata.json`.

The generated JSONL dataset is included at `data/public_conversations.jsonl`. Packed binaries are intentionally ignored because anyone can reproduce them with one command.

## Realistic expectations

This model is still tiny. It can learn frequent phrasing, basic conversational patterns, elementary arithmetic examples, and narrow coding templates. It cannot store broad world knowledge accurately, keep up with current events, reason like a frontier model, or reliably produce complex software. Expect hallucinations, repetition, context mistakes, and incorrect facts. Never rely on it for high-stakes advice.

The full 2,500-step base run processes roughly 123 million token positions after gradient accumulation. Actual time depends heavily on the GPU. A modern NVIDIA gaming GPU should take hours, not minutes; CPU training can take days. Around 6 GB of VRAM is a comfortable target with the supplied batch size, although lower-memory cards can use `batch_size: 2` and more accumulation.

## Setup

From the standalone repository in normal Command Prompt:

```cmd
cd /d C:\path\to\ChudGPT-Public
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, install the matching CUDA PyTorch build described at https://pytorch.org/get-started/locally/.

## Build and train

```cmd
python prepare.py
python train_public.py --device cuda
python finetune_public.py --device cuda
python evaluate_public.py --device cuda
python chat_public.py --device cuda
```

Resume interrupted stages:

```cmd
python train_public.py --device cuda --resume checkpoints\base\latest.pt
python finetune_public.py --device cuda --resume checkpoints\chat\latest.pt
```

CPU is supported by replacing `cuda` with `cpu`. `both` keeps the model on CUDA while CPU workers prepare batches.

## Export for Hugging Face

Only export after training and evaluation:

```cmd
python export_huggingface.py --checkpoint checkpoints\chat\best.pt
```

This creates `release` with safe tensor weights, tokenizer, model card, standalone architecture, and chat client. Test it before uploading:

```cmd
python release\inference.py --model-dir release
```

## Publish to Hugging Face

1. Create an account at https://huggingface.co/join.
2. In Hugging Face, open **Settings → Access Tokens** and create a token with write access.
3. Authenticate locally:

```cmd
hf auth login
```

4. Upload the release folder (replace the username):

```cmd
python upload_huggingface.py YOUR_USERNAME/ChudGPT-Public
```

The script creates the model repository and uploads the complete release. To create it privately first, add `--private`. You can also use the official CLI directly:

```cmd
hf upload YOUR_USERNAME/ChudGPT-Public release .
```

## Use it through an API with a Hugging Face token

A custom PyTorch model repository is downloadable but does not automatically become a hosted inference API. This project therefore includes a separate private Gradio Space in `space/`. Deploy it after uploading the model:

```cmd
python upload_space.py YOUR_USERNAME/ChudGPT-Public-API --model-repo YOUR_USERNAME/ChudGPT-Public
```

The Space is created **private**, so callers authenticate with an ordinary Hugging Face read token. After the Space finishes building:

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/ChudGPT-Public-API", token="hf_your_read_token")
answer = client.predict(
    message="What is 7 + 5?",
    history=[],
    api_name="/chat",
)
print(answer)
```

You can create a fine-grained read token at https://huggingface.co/settings/tokens. Keep it secret and never put it in browser JavaScript or commit it to GitHub. A public Space can be called without a token, although authenticated calls receive better limits. A private Space requires a token from an account allowed to access it.

Hugging Face model repositories support custom PyTorch models, so Transformers compatibility is not required. Users run the included `inference.py`; the model card clearly describes architecture, data, limitations, and usage.

## Important publishing rule

Do not upload an untrained or barely trained checkpoint while presenting it as finished. Complete training, inspect validation loss, read the evaluation responses, and update the Hugging Face model card with real metrics before publishing.
