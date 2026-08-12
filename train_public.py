"""Run ChudGPT-Public base training, optionally resuming a checkpoint."""

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
command = [sys.executable, "train.py", "--config", "configs/train.yaml", "--device", args.device]
if args.resume:
    command.extend(["--resume", args.resume])
subprocess.run(command, cwd=ROOT, check=True)
