from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .prompts import DEFAULT_SYSTEM_PROMPT, build_context_token_ids, normalize_messages


def load_sft_records(path: Path) -> list[list[dict[str, str]]]:
    """Load and validate instruction/response or multi-turn JSONL examples."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Fine-tuning dataset not found: {path}. See README.md for the JSONL format."
        )
    records: list[list[dict[str, str]]] = []
    fingerprints: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if "messages" in raw:
                    messages = normalize_messages(raw["messages"])
                else:
                    messages = normalize_messages(
                        [
                            {"role": "user", "content": raw["instruction"]},
                            {"role": "assistant", "content": raw["response"]},
                        ]
                    )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid fine-tuning record at {path}:{line_number}: {exc}") from exc
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                raise ValueError(
                    f"Record at {path}:{line_number} must end with an assistant response"
                )
            if not any(message["role"] == "user" for message in messages[:-1]):
                raise ValueError(f"Record at {path}:{line_number} has no user prompt")
            fingerprint = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            if fingerprint not in fingerprints:
                records.append(messages)
                fingerprints.add(fingerprint)
    if len(records) < 2:
        raise ValueError("Fine-tuning requires at least two unique valid examples")
    return records


class SupervisedConversationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Tokenize conversations and train only on the final assistant response."""

    def __init__(
        self,
        records: Sequence[Sequence[Mapping[str, object]]],
        tokenizer: Any,
        context_length: int,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        eos_id = tokenizer.token_to_id("<eos>")
        pad_id = tokenizer.token_to_id("<pad>")
        if eos_id is None or pad_id is None:
            raise ValueError("Tokenizer must contain <eos> and <pad> tokens")
        self.pad_id = int(pad_id)
        for messages in records:
            normalized = normalize_messages(messages)
            response = normalized[-1]["content"]
            prompt_messages = normalized[:-1]
            _, prompt_ids = build_context_token_ids(
                tokenizer, prompt_messages, context_length, system_prompt=system_prompt
            )
            response_ids = tokenizer.encode(f" {response}").ids + [int(eos_id)]
            available = context_length + 1 - len(prompt_ids)
            if available < 2:
                continue
            full_ids = prompt_ids + response_ids[:available]
            inputs = torch.tensor(full_ids[:-1], dtype=torch.long)
            targets = torch.tensor(full_ids[1:], dtype=torch.long)
            # The target aligned with the final prompt token predicts response token one.
            targets[: max(0, len(prompt_ids) - 1)] = -100
            self.examples.append((inputs, targets))
        if not self.examples:
            raise ValueError("No fine-tuning examples fit within the model context length")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]

    def collate(self, batch: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        maximum = max(inputs.numel() for inputs, _ in batch)
        inputs = torch.full((len(batch), maximum), self.pad_id, dtype=torch.long)
        targets = torch.full((len(batch), maximum), -100, dtype=torch.long)
        for index, (example_inputs, example_targets) in enumerate(batch):
            inputs[index, : example_inputs.numel()] = example_inputs
            targets[index, : example_targets.numel()] = example_targets
        return inputs, targets


def split_records(
    records: list[list[dict[str, str]]], validation_fraction: float, seed: int
) -> tuple[list[list[dict[str, str]]], list[list[dict[str, str]]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    if validation_count >= len(shuffled):
        validation_count = len(shuffled) - 1
    return shuffled[validation_count:], shuffled[:validation_count]
