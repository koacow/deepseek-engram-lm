#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engram_lm.train import TrainConfig, train


if __name__ == "__main__":
    train(
        TrainConfig(
            model_kind="params",
        )
    )
