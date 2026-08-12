"""Export a trained project checkpoint as a safe Hugging Face model folder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/chat/best.pt")
    parser.add_argument("--output", default="release")
    args = parser.parse_args()
    checkpoint_path = ROOT / args.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint_path}")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("model.py", "inference.py", "requirements.txt", "README.md", "LICENSE"):
        shutil.copy2(ROOT / "hf" / filename, output / filename)
    shutil.copy2(ROOT / "artifacts" / "tokenizer.json", output / "tokenizer.json")
    config = json.loads((ROOT / "hf" / "config.json").read_text(encoding="utf-8"))
    config["model_config"] = saved["model_config"]
    config["training_step"] = int(saved.get("step", 0))
    config["validation_loss"] = saved.get("best_validation_loss")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Clone tensors because the token embedding and language-model head are
    # deliberately tied in memory; safetensors requires independent storage.
    tensors = {key: value.detach().contiguous().clone() for key, value in saved["model"].items()}
    save_file(tensors, output / "model.safetensors", metadata={"format": "pt"})
    print(f"Hugging Face release exported to {output}")


if __name__ == "__main__":
    main()
