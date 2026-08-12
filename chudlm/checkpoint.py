from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically save a checkpoint, tolerating short Windows/OneDrive locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(state, temporary)
        for attempt in range(20):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.25 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # ChudGPT checkpoints contain tensors and primitive dictionaries only.
    # Restricted loading prevents a replaced/untrusted checkpoint from using
    # pickle to construct arbitrary Python objects during startup.
    return torch.load(path, map_location=device, weights_only=True)
