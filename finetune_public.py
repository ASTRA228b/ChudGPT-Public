"""Run response-only conversational fine-tuning for ChudGPT-Public."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["auto", "cpu", "cuda", "both"], default="auto")
parser.add_argument("--resume")
args = parser.parse_args()
command = [sys.executable, "fine_tune.py", "--config", "configs/finetune.yaml", "--device", args.device]
if args.resume:
    command.extend(["--resume", args.resume])
subprocess.run(command, cwd=ROOT, check=True)
