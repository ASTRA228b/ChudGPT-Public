from __future__ import annotations

import argparse
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from chudlm.checkpoint import load_checkpoint, save_checkpoint
from chudlm.config import dataclass_from_dict, load_yaml
from chudlm.data import TokenBlockDataset
from chudlm.model import ModelConfig, TransformerLM


@dataclass
class TrainConfig:
    model_config: str = "configs/model.yaml"
    data_dir: str = "data/processed"
    output_dir: str = "checkpoints"
    seed: int = 42
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    epochs: int = 3
    warmup_steps: int = 200
    max_steps: int | None = None
    gradient_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 50
    save_interval: int = 500
    num_workers: int = 0
    amp: bool = True
    compile: bool = False
    resume_from: str | None = None
    initialize_from: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the conversational transformer.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint path; overrides resume_from in YAML.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "both"], default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested in {"cuda", "both"} and not torch.cuda.is_available():
        raise RuntimeError(f"{requested} was requested but CUDA is unavailable")
    use_cuda = requested in {"cuda", "both"} or (
        requested == "auto" and torch.cuda.is_available()
    )
    if requested == "both":
        print("Both mode: CUDA runs the model; CPU workers load and prepare batches.")
    return torch.device("cuda" if use_cuda else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer: AdamW, warmup: int, total: int, minimum_ratio: float) -> LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup:
            return max(1e-8, step / max(1, warmup))
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, multiplier)


@torch.inference_mode()
def evaluate(model: TransformerLM, loader: DataLoader[Any], device: torch.device, batches: int, amp: bool) -> float:
    model.eval()
    losses: list[float] = []
    context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp and device.type == "cuda" else nullcontext()
    for index, (inputs, targets) in enumerate(loader):
        if index >= batches:
            break
        with context:
            _, loss = model(inputs.to(device), targets.to(device))
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def checkpoint_state(model: TransformerLM, optimizer: AdamW, scheduler: LambdaLR, scaler: Any, step: int, epoch: int, best: float) -> dict[str, Any]:
    unwrapped_model = getattr(model, "_orig_mod", model)
    return {
        "model": unwrapped_model.state_dict(), "model_config": unwrapped_model.config_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step, "epoch": epoch,
        "best_validation_loss": best,
    }


def main() -> None:
    args = parse_args()
    config = dataclass_from_dict(TrainConfig, load_yaml(args.config))
    model_config = dataclass_from_dict(ModelConfig, load_yaml(config.model_config))
    if config.gradient_accumulation_steps < 1 or config.batch_size < 1:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    seed_everything(config.seed)
    device = select_device(args.device)
    amp_enabled = config.amp and device.type == "cuda"
    data_dir = Path(config.data_dir)
    train_data = TokenBlockDataset(data_dir / "train.bin", model_config.context_length, config.seed)
    validation_data = TokenBlockDataset(data_dir / "validation.bin", model_config.context_length, config.seed + 1)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=config.batch_size, num_workers=config.num_workers)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = config.max_steps or updates_per_epoch * config.epochs

    model = TransformerLM(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.95))
    scheduler = make_scheduler(optimizer, config.warmup_steps, total_steps, config.min_learning_rate / config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    step, start_epoch, best_validation = 0, 0, float("inf")
    resume_path = args.resume or config.resume_from
    if resume_path and config.initialize_from:
        raise ValueError("Use either resume_from or initialize_from, not both")
    if resume_path:
        saved = load_checkpoint(Path(resume_path), device)
        if saved["model_config"] != model.config_dict():
            raise ValueError("Checkpoint model configuration does not match the current model configuration")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved.get("scaler", {}))
        step, start_epoch = int(saved["step"]), int(saved["epoch"])
        best_validation = float(saved.get("best_validation_loss", best_validation))
        print(f"Resumed {resume_path} at step {step}, epoch {start_epoch}")
    elif config.initialize_from:
        saved = load_checkpoint(Path(config.initialize_from), device)
        if saved["model_config"] != model.config_dict():
            raise ValueError("Initialization checkpoint model configuration does not match")
        model.load_state_dict(saved["model"])
        print(f"Initialized model weights from {config.initialize_from}; optimizer and schedule start fresh")

    if config.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    output_dir = Path(config.output_dir)
    print(f"Training {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters on {device}; AMP={amp_enabled}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss, running_batches, running_start = 0.0, 0, time.time()
    for epoch in range(start_epoch, config.epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for batch_index, (inputs, targets) in enumerate(progress):
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with context:
                _, loss = model(inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True))
                assert loss is not None
                scaled_loss = loss / config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            running_loss += loss.item()
            running_batches += 1
            should_update = (batch_index + 1) % config.gradient_accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if not should_update:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step % config.log_interval == 0:
                elapsed = max(time.time() - running_start, 1e-6)
                progress.set_postfix(loss=f"{running_loss / max(1, running_batches):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}", steps_s=f"{config.log_interval / elapsed:.2f}")
                running_loss, running_batches, running_start = 0.0, 0, time.time()
            if step % config.eval_interval == 0:
                validation_loss = evaluate(model, validation_loader, device, config.eval_batches, amp_enabled)
                print(f"\nStep {step}: validation loss {validation_loss:.4f}, perplexity {math.exp(min(validation_loss, 20)):.2f}")
                if validation_loss < best_validation:
                    best_validation = validation_loss
                    save_checkpoint(output_dir / "best.pt", checkpoint_state(model, optimizer, scheduler, scaler, step, epoch, best_validation))
            if step % config.save_interval == 0:
                save_checkpoint(output_dir / f"step_{step}.pt", checkpoint_state(model, optimizer, scheduler, scaler, step, epoch, best_validation))
            if step >= total_steps:
                break
        save_checkpoint(output_dir / "latest.pt", checkpoint_state(model, optimizer, scheduler, scaler, step, epoch + 1, best_validation))
        if step >= total_steps:
            break
    print(f"Training complete at step {step}. Latest checkpoint: {output_dir / 'latest.pt'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
