from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from chudlm.checkpoint import load_checkpoint, save_checkpoint
from chudlm.config import dataclass_from_dict, load_yaml
from chudlm.model import ModelConfig, TransformerLM
from chudlm.sft_data import SupervisedConversationDataset, load_sft_records, split_records
from train import make_scheduler, seed_everything, select_device


@dataclass
class FineTuneConfig:
    model_config: str = "configs/model.yaml"
    tokenizer_path: str = "artifacts/tokenizer.json"
    dataset_path: str = "data/finetune/conversations.jsonl"
    base_checkpoint: str = "checkpoints/latest.pt"
    output_dir: str = "checkpoints/finetuned"
    seed: int = 42
    validation_fraction: float = 0.05
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    min_learning_rate: float = 5e-6
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_steps: int = 50
    max_steps: int | None = None
    gradient_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 50
    save_interval: int = 250
    num_workers: int = 0
    amp: bool = True
    resume_from: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tune ChudGPT on answer examples.")
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "both"], default="auto")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(
    model: TransformerLM,
    loader: DataLoader[Any],
    device: torch.device,
    maximum_batches: int,
    amp_enabled: bool,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        context = (
            torch.autocast("cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with context:
            _, loss = model(inputs.to(device), targets.to(device))
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def state(
    model: TransformerLM,
    optimizer: AdamW,
    scheduler: Any,
    scaler: Any,
    step: int,
    epoch: int,
    best_loss: float,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "model_config": model.config_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_validation_loss": best_loss,
        "training_stage": "supervised_fine_tuning",
    }


def main() -> None:
    args = parse_args()
    config = dataclass_from_dict(FineTuneConfig, load_yaml(args.config))
    model_config = dataclass_from_dict(ModelConfig, load_yaml(config.model_config))
    seed_everything(config.seed)
    device = select_device(args.device)
    amp_enabled = config.amp and device.type == "cuda"
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match model vocab_size")

    records = load_sft_records(Path(config.dataset_path))
    training_records, validation_records = split_records(
        records, config.validation_fraction, config.seed
    )
    training_data = SupervisedConversationDataset(
        training_records, tokenizer, model_config.context_length
    )
    validation_data = SupervisedConversationDataset(
        validation_records, tokenizer, model_config.context_length
    )
    training_loader = DataLoader(
        training_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=training_data.collate,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=validation_data.collate,
    )
    updates_per_epoch = math.ceil(
        len(training_loader) / config.gradient_accumulation_steps
    )
    total_steps = config.max_steps or updates_per_epoch * config.epochs

    model = TransformerLM(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = make_scheduler(
        optimizer,
        config.warmup_steps,
        total_steps,
        config.min_learning_rate / config.learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    step, start_epoch, best_validation = 0, 0, float("inf")
    resume_path = args.resume or config.resume_from
    source_path = Path(resume_path or config.base_checkpoint)
    saved = load_checkpoint(source_path, device)
    if saved["model_config"] != model.config_dict():
        raise ValueError("Checkpoint model configuration does not match configs/model.yaml")
    model.load_state_dict(saved["model"])
    if resume_path:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved.get("scaler", {}))
        step = int(saved["step"])
        start_epoch = int(saved["epoch"])
        best_validation = float(saved.get("best_validation_loss", best_validation))
        print(f"Resumed fine-tuning from {source_path} at step {step}")
    else:
        print(f"Initialized fine-tuning from base checkpoint {source_path}")

    print(
        f"Fine-tuning {len(training_data):,} examples; validating on "
        f"{len(validation_data):,}; device={device}; AMP={amp_enabled}"
    )
    output_dir = Path(config.output_dir)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss, running_batches, started = 0.0, 0, time.perf_counter()
    for epoch in range(start_epoch, config.epochs):
        progress = tqdm(training_loader, desc=f"Fine-tune {epoch + 1}/{config.epochs}")
        for batch_index, (inputs, targets) in enumerate(progress):
            context = (
                torch.autocast("cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with context:
                _, loss = model(
                    inputs.to(device, non_blocking=True),
                    targets.to(device, non_blocking=True),
                )
                assert loss is not None
                accumulated_loss = loss / config.gradient_accumulation_steps
            scaler.scale(accumulated_loss).backward()
            running_loss += loss.item()
            running_batches += 1
            update = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(training_loader)
            )
            if not update:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step % config.log_interval == 0:
                elapsed = max(time.perf_counter() - started, 1e-6)
                progress.set_postfix(
                    loss=f"{running_loss / max(1, running_batches):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    steps_s=f"{config.log_interval / elapsed:.2f}",
                )
                running_loss, running_batches, started = 0.0, 0, time.perf_counter()
            if step % config.eval_interval == 0:
                validation_loss = evaluate(
                    model, validation_loader, device, config.eval_batches, amp_enabled
                )
                print(f"\nStep {step}: response-only validation loss {validation_loss:.4f}")
                if validation_loss < best_validation:
                    best_validation = validation_loss
                    save_checkpoint(
                        output_dir / "best.pt",
                        state(model, optimizer, scheduler, scaler, step, epoch, best_validation),
                    )
            if step % config.save_interval == 0:
                save_checkpoint(
                    output_dir / f"step_{step}.pt",
                    state(model, optimizer, scheduler, scaler, step, epoch, best_validation),
                )
            if step >= total_steps:
                break
        epoch_validation_loss = evaluate(
            model, validation_loader, device, config.eval_batches, amp_enabled
        )
        print(
            f"Epoch {epoch + 1}: response-only validation loss "
            f"{epoch_validation_loss:.4f}"
        )
        if epoch_validation_loss < best_validation:
            best_validation = epoch_validation_loss
            save_checkpoint(
                output_dir / "best.pt",
                state(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    epoch + 1,
                    best_validation,
                ),
            )
        save_checkpoint(
            output_dir / "latest.pt",
            state(model, optimizer, scheduler, scaler, step, epoch + 1, best_validation),
        )
        if step >= total_steps:
            break
    print(f"Fine-tuning complete. Checkpoint: {output_dir / 'latest.pt'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:
        print(f"Fine-tuning failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
