#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engram_lm.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Engram variant")
    parser.add_argument("--frozen-gates", action="store_true", help="Use fixed alpha_t = 0.5 gates")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train(
        TrainConfig(
            model_kind="engram",
            engram_gate_frozen=args.frozen_gates,
            steps=args.steps,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
