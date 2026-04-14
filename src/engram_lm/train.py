from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import collate_lm_batch, load_wikitext103
from .modeling import BaselineLM, EngramLM, ExperimentConfig, ParamsControlLM


ModelKind = Literal["baseline", "params", "engram"]


@dataclass
class TrainConfig:
    model_kind: ModelKind
    output_dir: str = "checkpoints"
    tokenizer_name: str = "gpt2"
    block_size: int = 512
    # Effective batch = batch_size × grad_accum_steps.
    # Default gives effective batch = 8 × 4 = 32, matching the proposal.
    batch_size: int = 8
    grad_accum_steps: int = 4
    steps: int = 100_000
    warmup_steps: int = 2_000
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    engram_gate_frozen: bool = False
    # Evaluation and checkpointing
    eval_interval: int = 5_000  # evaluate on val set every N optimizer steps
    save_interval: int = 10_000  # save checkpoint every N optimizer steps
    log_interval: int = 100     # log training loss every N optimizer steps
    seed: int = 42
    resume_from: Optional[str] = None  # path to checkpoint to resume from


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(exp_cfg: ExperimentConfig, model_kind: ModelKind, pad_id: int) -> nn.Module:
    if model_kind == "baseline":
        return BaselineLM(exp_cfg)
    if model_kind == "params":
        return ParamsControlLM(exp_cfg)
    if model_kind == "engram":
        return EngramLM(exp_cfg, pad_id=pad_id)
    raise ValueError(f"Unknown model kind: {model_kind}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup followed by cosine decay to min_lr."""
    warmup = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=min_lr)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, max_batches: int = 200) -> float:
    """Compute perplexity on a dataloader (up to max_batches for speed).

    Returns perplexity = exp(mean cross-entropy loss).
    Uses the same causal-shifted loss as training.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        if total_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, labels=labels)
        total_loss += loss.item()
        total_batches += 1
    model.train()
    avg_loss = total_loss / max(1, total_batches)
    return float(torch.exp(torch.tensor(avg_loss)).item())


def _save_checkpoint(
    path: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_cfg: "TrainConfig",
    exp_cfg: ExperimentConfig,
) -> None:
    torch.save(
        {
            "step": step,
            "model_kind": train_cfg.model_kind,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "exp_cfg": exp_cfg.__dict__,
            "tokenizer_name": train_cfg.tokenizer_name,
        },
        path,
    )


def _load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
) -> int:
    """Load checkpoint and return the step to resume from."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt["step"])


def train(train_cfg: TrainConfig) -> nn.Module:
    set_seed(train_cfg.seed)

    datasets, tokenizer_bundle = load_wikitext103(train_cfg.tokenizer_name, train_cfg.block_size)
    exp_cfg = ExperimentConfig(
        vocab_size=tokenizer_bundle.vocab_size,
        block_size=train_cfg.block_size,
        engram_gate_frozen=train_cfg.engram_gate_frozen,
    )
    model = build_model(exp_cfg, train_cfg.model_kind, tokenizer_bundle.pad_id).to(train_cfg.device)

    # --- Data loaders ---
    train_loader = DataLoader(
        datasets["train"],
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_lm_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        datasets["validation"],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_lm_batch,
        pin_memory=torch.cuda.is_available(),
    )

    # --- Optimizer + LR scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=train_cfg.warmup_steps,
        total_steps=train_cfg.steps,
        min_lr=train_cfg.min_lr,
    )

    # --- Output directory + log file ---
    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{train_cfg.model_kind}_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_ppl", "lr"])

    # --- Optionally resume from checkpoint ---
    start_step = 0
    if train_cfg.resume_from is not None:
        start_step = _load_checkpoint(train_cfg.resume_from, model, optimizer, scheduler, train_cfg.device)
        print(f"Resumed from {train_cfg.resume_from} at step {start_step}")

    # --- Training loop ---
    model.train()
    step = start_step
    accum_loss = 0.0  # accumulated loss over grad_accum_steps micro-steps
    micro_step = 0    # micro-step counter within current optimizer step
    recent_losses: list[float] = []  # for rolling log

    progress = tqdm(total=train_cfg.steps, initial=start_step, desc=f"training-{train_cfg.model_kind}")
    data_iter = iter(train_loader)

    while step < train_cfg.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(train_cfg.device)
        labels = batch["labels"].to(train_cfg.device)

        # Forward + backward (gradient accumulation)
        _, loss = model(input_ids, labels=labels)
        # Scale loss so that gradients average correctly across accumulation steps
        (loss / train_cfg.grad_accum_steps).backward()
        accum_loss += loss.item()
        micro_step += 1

        if micro_step < train_cfg.grad_accum_steps:
            continue  # accumulate more gradients before updating

        # --- Optimizer step ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        avg_step_loss = accum_loss / train_cfg.grad_accum_steps
        recent_losses.append(avg_step_loss)
        if len(recent_losses) > 100:
            recent_losses.pop(0)

        accum_loss = 0.0
        micro_step = 0
        step += 1
        progress.update(1)
        progress.set_postfix(
            loss=f"{sum(recent_losses) / len(recent_losses):.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        # --- Logging ---
        if step % train_cfg.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, avg_step_loss, "", f"{current_lr:.6e}"])

        # --- Validation (PPL) ---
        if step % train_cfg.eval_interval == 0:
            val_ppl = evaluate(model, val_loader, train_cfg.device)
            current_lr = scheduler.get_last_lr()[0]
            progress.write(f"[step {step:>6}] val_ppl={val_ppl:.2f}  lr={current_lr:.2e}")
            # Back-fill val_ppl into the CSV log row for this step
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, avg_step_loss, f"{val_ppl:.4f}", f"{current_lr:.6e}"])

        # --- Periodic checkpoint save ---
        if step % train_cfg.save_interval == 0:
            ckpt_path = out_dir / f"{train_cfg.model_kind}_step{step:07d}.pt"
            _save_checkpoint(ckpt_path, step, model, optimizer, scheduler, train_cfg, exp_cfg)
            progress.write(f"Checkpoint saved: {ckpt_path}")

    progress.close()

    # --- Final checkpoint ---
    final_path = out_dir / f"{train_cfg.model_kind}_final.pt"
    _save_checkpoint(final_path, step, model, optimizer, scheduler, train_cfg, exp_cfg)
    print(f"Training complete. Final checkpoint: {final_path}")

    # --- Final evaluation on test split ---
    test_loader = DataLoader(
        datasets["test"],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_lm_batch,
        pin_memory=torch.cuda.is_available(),
    )
    test_ppl = evaluate(model, test_loader, train_cfg.device, max_batches=500)
    print(f"Test PPL: {test_ppl:.2f}")
    # Persist test results alongside checkpoint
    results_path = out_dir / f"{train_cfg.model_kind}_results.json"
    with open(results_path, "w") as f:
        json.dump({"model_kind": train_cfg.model_kind, "test_ppl": test_ppl, "steps": step}, f, indent=2)

    return model
