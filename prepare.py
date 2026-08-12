"""Build the public dataset and train its tokenizer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    run([sys.executable, "build_public_data.py"])
    public_tokenizer = ROOT / "artifacts" / "tokenizer.json"
    public_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    bundled_tokenizer = ROOT / "artifacts" / "base_tokenizer.json"
    shutil.copy2(bundled_tokenizer, public_tokenizer)
    print(f"Copied the bundled natural-language tokenizer to {public_tokenizer}")
    run([
        sys.executable, "prepare_data.py",
        "--config", "configs/data.yaml",
        "--model-config", "configs/model.yaml",
    ])
