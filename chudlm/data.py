from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from .prompts import format_conversation


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def iter_documents(path: Path, text_field: str, dialogue_field: str) -> Iterator[str]:
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.suffix.lower() in {".txt", ".jsonl"}
    )
    if not files:
        raise FileNotFoundError(f"No .txt or .jsonl files found at {path}")
    for file_path in files:
        if file_path.suffix.lower() == ".txt":
            yield file_path.read_text(encoding="utf-8", errors="replace")
            continue
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if dialogue_field in record:
                        yield format_conversation(record[dialogue_field])
                    else:
                        yield str(record[text_field])
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"Invalid record in {file_path}:{line_number}: {exc}") from exc


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: Path, context_length: int, seed: int = 42) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Token file not found: {path}. Run prepare_data.py first.")
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        self.seed = seed
        if len(self.tokens) <= context_length:
            raise ValueError(f"{path} has {len(self.tokens)} tokens; need more than {context_length}")

    def __len__(self) -> int:
        return max(1, (len(self.tokens) - 1) // self.context_length)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + index)
        start = rng.randrange(0, len(self.tokens) - self.context_length)
        chunk = np.asarray(self.tokens[start : start + self.context_length + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())
