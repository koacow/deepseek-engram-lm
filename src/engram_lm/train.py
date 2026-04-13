from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
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
    batch_size: int = 4
    steps: int = 1000
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    engram_gate_frozen: bool = False


def build_model(exp_cfg: ExperimentConfig, model_kind: ModelKind, pad_id: int) -> torch.nn.Module:
    if model_kind == "baseline":
        return BaselineLM(exp_cfg)
    if model_kind == "params":
        return ParamsControlLM(exp_cfg)
    if model_kind == "engram":
        return EngramLM(exp_cfg, pad_id=pad_id)
    raise ValueError(f"Unknown model kind: {model_kind}")


def train(train_cfg: TrainConfig):
    datasets, tokenizer_bundle = load_wikitext103(train_cfg.tokenizer_name, train_cfg.block_size)
    exp_cfg = ExperimentConfig(
        vocab_size=tokenizer_bundle.vocab_size,
        block_size=train_cfg.block_size,
        engram_gate_frozen=train_cfg.engram_gate_frozen,
    )
    model = build_model(exp_cfg, train_cfg.model_kind, tokenizer_bundle.pad_id).to(train_cfg.device)

    train_ds = datasets["train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_lm_batch,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    model.train()
    step = 0
    running_loss = 0.0
    progress = tqdm(total=train_cfg.steps, desc=f"training-{train_cfg.model_kind}")
    data_iter = iter(train_loader)

    while step < train_cfg.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(train_cfg.device)
        labels = batch["labels"].to(train_cfg.device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        running_loss += float(loss.item())
        step += 1
        progress.update(1)
        progress.set_postfix(loss=f"{running_loss / step:.4f}")

    progress.close()

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": train_cfg.model_kind,
            "model_state_dict": model.state_dict(),
            "exp_cfg": exp_cfg.__dict__,
            "tokenizer_name": train_cfg.tokenizer_name,
        },
        out_dir / f"{train_cfg.model_kind}.pt",
    )
    return model
