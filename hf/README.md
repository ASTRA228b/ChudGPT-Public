---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- pytorch
- custom-code
- conversational
- experimental
---

# ChudGPT-Public 21M

ChudGPT-Public is a small, experimental, decoder-only conversational language model with **20,999,184 trainable parameters**. It is an independent ChudGPT variant intended for learning, local experiments, and low-resource inference.

Created by [ASTRA228b](https://github.com/ASTRA228b).

## Architecture

- Vocabulary: 8,192 byte-level BPE tokens
- Context: 1,024 tokens
- Embedding width: 384
- Layers: 9
- Attention heads: 6
- Feed-forward width: 1,808
- Positional encoding: RoPE
- Parameters: 20,999,184

## Dataset

The supplied training pipeline builds 20,000 project-authored conversations containing 60,000 messages. Topics include model self-knowledge, general information, geography, basic science, computing concepts, arithmetic, short stories, jokes, emotional support, Python, JavaScript, C#, and Unity. No live internet or retrieval system is included.

## Run

```bash
git clone https://huggingface.co/YOUR_USERNAME/ChudGPT-Public
cd ChudGPT-Public
python -m pip install -r requirements.txt
python inference.py --model-dir .
```

Or download automatically:

```bash
python inference.py --repo-id YOUR_USERNAME/ChudGPT-Public
```

## Limitations

This is a tiny model trained from scratch. It will not perform like ChatGPT or modern billion-parameter assistants. It may hallucinate, repeat text, misunderstand prompts, produce incorrect code, and give unreliable factual answers. Do not use it for medical, legal, financial, safety-critical, or security decisions. Verify important information independently.

## Files

- `model.safetensors`: trained weights (added by the export script)
- `config.json`: architecture configuration
- `tokenizer.json`: byte-level BPE tokenizer
- `model.py`: standalone PyTorch architecture
- `inference.py`: terminal chat client
