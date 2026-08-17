"""Command-line entry point for pretraining and exact resume."""

from __future__ import annotations

import argparse
from pathlib import Path

from mllms.training.config import TrainingConfig, load_training_config
from mllms.training.engine import train


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--p-clone", type=float)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--target-seen-tokens", type=int)
    parser.add_argument("--target-epochs", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    values = vars(parser.parse_args())

    config_path = values.pop("config")
    config = (
        load_training_config(config_path)
        if config_path is not None
        else TrainingConfig()
    )
    overrides = {key: value for key, value in values.items() if value is not None}
    if not overrides:
        return config
    merged = {**vars(config), **overrides}
    return TrainingConfig(**merged)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
