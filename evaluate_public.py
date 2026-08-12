"""Evaluate a ChudGPT-Public checkpoint on the standard unseen prompt set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/chat/best.pt")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
args = parser.parse_args()
subprocess.run([
    sys.executable, "evaluate_checkpoint.py", "--checkpoint", args.checkpoint,
    "--tokenizer", "artifacts/tokenizer.json",
    "--output", "reports/evaluation.json", "--device", args.device,
], cwd=ROOT, check=True)
