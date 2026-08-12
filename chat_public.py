"""Open local chat using the trained public checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/chat/best.pt")
parser.add_argument("--device", choices=["auto", "cpu", "cuda", "both"], default="auto")
args = parser.parse_args()
subprocess.run([
    sys.executable, "chat.py", "--checkpoint", args.checkpoint,
    "--tokenizer", "artifacts/tokenizer.json",
    "--max-new-tokens", "160", "--temperature", "0.7", "--top-k", "50",
    "--top-p", "0.9", "--repetition-penalty", "1.15", "--device", args.device,
], cwd=ROOT, check=True)
