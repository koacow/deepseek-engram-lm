#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engram_lm.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the iso-parameter control transformer")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (effective batch = batch-size × grad-accum)")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        TrainConfig(
            model_kind="params",
            steps=args.steps,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            output_dir=args.output_dir,
            resume_from=args.resume_from,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
