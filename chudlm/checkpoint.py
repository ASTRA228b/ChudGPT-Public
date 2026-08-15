from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically save a checkpoint, tolerating short Windows/OneDrive locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # OneDrive may open a partially written archive and lock it. Stage the
    # complete checkpoint outside the synchronized workspace, then atomically
    # replace the destination in one operation.
    staging_root = Path(tempfile.gettempdir()) / "chudgpt-checkpoints"
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = staging_root / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(state, temporary)
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.25 * (attempt + 1))
    finally:
        for attempt in range(10):
            try:
                temporary.unlink(missing_ok=True)
                break
            except PermissionError:
                if attempt == 9:
                    break
                time.sleep(0.1 * (attempt + 1))


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # ChudGPT checkpoints contain tensors and primitive dictionaries only.
    # Restricted loading prevents a replaced/untrusted checkpoint from using
    # pickle to construct arbitrary Python objects during startup.
    return torch.load(path, map_location=device, weights_only=True)
