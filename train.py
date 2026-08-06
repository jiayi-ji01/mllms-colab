"""Train the GPT model on balanced original and cloned token streams."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml

from data_lib.cloned import ClonedMapper, TokenStream
from model.config import GPTConfig
from model.model import GPT
from tokenizer.tokenizer import load_tokenizer


@dataclass
class TrainingConfig:
    """Model, data, optimization, and runtime settings."""

    data_dir: str = "data/processed"
    tokenizer_path: str = "artifacts/tokenizer/tokenizer.model"
    output_dir: str = "outputs/gpt12_tinystories_clone"
    resume: str | None = None
    device: str = "auto"
    seed: int = 42
    p_clone: float = 0.5

    # This is the SentencePiece/base vocabulary size. ClonedMapper creates a
    # model vocabulary twice this size so original and clone IDs stay disjoint.
    vocab_size: int = 4096
    block_size: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 12
    d_ff: int = 1024
    dropout: float = 0.1
    bias: bool = True

    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    target_seen_tokens: int = 200_000_000
    max_steps: int | None = None
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_ratio: float = 0.01
    warmup_steps: int | None = None
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 20
    checkpoint_interval: int = 1_000
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "target_seen_tokens": self.target_seen_tokens,
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
            "log_interval": self.log_interval,
            "eval_interval": self.eval_interval,
            "eval_batches": self.eval_batches,
            "checkpoint_interval": self.checkpoint_interval,
        }
        if self.max_steps is not None:
            positive["max_steps"] = self.max_steps
        if self.warmup_steps is not None:
            positive["warmup_steps"] = self.warmup_steps
        if self.early_stopping_patience is not None:
            positive["early_stopping_patience"] = self.early_stopping_patience
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not 0.0 <= self.p_clone <= 1.0:
            raise ValueError("p_clone must be between 0 and 1")
        if not 0.0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError(
                "min_learning_rate must be between 0 and learning_rate"
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must satisfy 0 <= warmup_ratio < 1")
        if self.warmup_steps is not None and self.warmup_ratio != 0.0:
            raise ValueError("set either warmup_steps or warmup_ratio, not both")

    @property
    def tokens_per_step(self) -> int:
        """Number of training tokens processed by one optimizer step."""
        return (
            self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.block_size
        )

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def resolved_max_steps(self) -> int:
        if self.max_steps is not None:
            return self.max_steps
        return math.ceil(self.target_seen_tokens / self.tokens_per_step)

    @property
    def resolved_warmup_steps(self) -> int:
        if self.warmup_steps is not None:
            return self.warmup_steps
        return math.ceil(self.warmup_ratio * self.resolved_max_steps)


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def _select_precision(device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def _make_grad_scaler(enabled: bool):
    """Create a CUDA GradScaler across supported PyTorch AMP APIs."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = config.resolved_warmup_steps
    max_steps = config.resolved_max_steps
    min_ratio = config.min_learning_rate / config.learning_rate

    if warmup_steps >= max_steps:
        raise ValueError("warmup steps must be smaller than total optimizer steps")

    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_span = max(1, max_steps - warmup_steps - 1)
        progress = min(1.0, (step - warmup_steps) / decay_span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


@torch.no_grad()
def evaluate(
    model: GPT,
    validation_stream: TokenStream,
    clone_mapper: ClonedMapper,
    config: TrainingConfig,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    """Evaluate original and clone loss on identical validation windows."""
    was_training = model.training
    model.eval()
    metrics: dict[str, float] = {}

    for language_id, name in ((0, "original"), (1, "clone")):
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        losses = []
        for _ in range(config.eval_batches):
            x, y, _ = validation_stream.get_batch(
                batch_size=config.micro_batch_size,
                block_size=config.block_size,
                device=device,
                cloned_mapper=clone_mapper,
                language=language_id,
                generator=generator,
            )
            with _autocast_context(device, precision):
                _, loss = model(x, targets=y)
            if loss is None:
                raise RuntimeError("model did not return validation loss")
            losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        metrics[f"{name}_loss"] = mean_loss
        metrics[f"{name}_perplexity"] = math.exp(mean_loss)

    metrics["average_loss"] = 0.5 * (
        metrics["original_loss"] + metrics["clone_loss"]
    )
    if was_training:
        model.train()
    return metrics


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    step: int,
    model_config: GPTConfig,
    training_config: TrainingConfig,
    best_validation_loss: float,
    tokens_seen: int,
    original_tokens_seen: int,
    clone_tokens_seen: int,
    evaluations_without_improvement: int,
    generator: torch.Generator,
) -> None:
    """Save all state required for an exact training resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "best_validation_loss": best_validation_loss,
        "tokens_seen": tokens_seen,
        "original_tokens_seen": original_tokens_seen,
        "clone_tokens_seen": clone_tokens_seen,
        "evaluations_without_improvement": evaluations_without_improvement,
        "generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator,
    expected_model_config: GPTConfig,
    device: torch.device,
) -> tuple[int, float, int, int, int, int]:
    """Restore model, optimizer, scheduler, scaler, counters, and RNG state."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"scheduler", "scaler", "tokens_seen"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(
            "Checkpoint predates the 12-layer Colab training format and cannot "
            f"be resumed; missing fields: {sorted(missing)}"
        )
    if checkpoint.get("model_config") != asdict(expected_model_config):
        raise ValueError(
            "Checkpoint model_config does not match this run. Use a new output "
            "directory and do not resume the old 4-layer checkpoint."
        )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    generator.set_state(checkpoint["generator_state"].cpu())
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
        cuda_rng_states = [
            state.cpu() for state in checkpoint["cuda_rng_state_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng_states)

    return (
        int(checkpoint["step"]),
        float(checkpoint["best_validation_loss"]),
        int(checkpoint["tokens_seen"]),
        int(checkpoint["original_tokens_seen"]),
        int(checkpoint["clone_tokens_seen"]),
        int(checkpoint["evaluations_without_improvement"]),
    )


def _write_log(log_file, record: dict[str, Any]) -> None:
    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    log_file.flush()


def train(config: TrainingConfig) -> None:
    """Run accumulated mixed-precision training and checkpointing."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = _select_device(config.device)
    precision = _select_precision(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    tokenizer = load_tokenizer(Path(config.tokenizer_path))
    if tokenizer.vocab_size() != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary is {tokenizer.vocab_size()}, but config "
            f"requires {config.vocab_size}"
        )

    clone_mapper = ClonedMapper(
        original_vocab_size=config.vocab_size,
        p_clone=config.p_clone,
        pad_id=tokenizer.pad_id(),
        generator=generator,
    )
    train_stream = TokenStream(config.data_dir, "train")
    validation_stream = TokenStream(config.data_dir, "validation")

    model_config = GPTConfig(
        vocab_size=clone_mapper.model_vocab_size,
        block_size=config.block_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        bias=config.bias,
    )
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, config)
    scaler = _make_grad_scaler(enabled=precision == "fp16")

    start_step = 0
    best_validation_loss = float("inf")
    tokens_seen = 0
    original_tokens_seen = 0
    clone_tokens_seen = 0
    evaluations_without_improvement = 0
    if config.resume is not None:
        (
            start_step,
            best_validation_loss,
            tokens_seen,
            original_tokens_seen,
            clone_tokens_seen,
            evaluations_without_improvement,
        ) = load_checkpoint(
            Path(config.resume),
            model,
            optimizer,
            scheduler,
            scaler,
            generator,
            model_config,
            device,
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    (output_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "model": asdict(model_config),
                "training": asdict(config),
                "resolved_max_steps": config.resolved_max_steps,
                "resolved_warmup_steps": config.resolved_warmup_steps,
                "tokens_per_step": config.tokens_per_step,
                "actual_train_dataset_tokens": len(train_stream.tokens),
                "parameter_count": parameter_count,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Device / precision: {device} / {precision}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Base / model vocabulary: {config.vocab_size:,} / {model_config.vocab_size:,}")
    print(f"Train dataset tokens (actual tokenized count): {len(train_stream.tokens):,}")
    print(f"Effective batch size: {config.effective_batch_size:,} sequences")
    print(f"Tokens per optimizer step: {config.tokens_per_step:,}")
    print(f"Optimizer steps: {config.resolved_max_steps:,}")
    print(f"Planned tokens seen: {config.resolved_max_steps * config.tokens_per_step:,}")
    print(f"Nominal epochs: {config.target_seen_tokens / len(train_stream.tokens):.3f}")
    print(f"Warmup steps: {config.resolved_warmup_steps:,}")
    print(f"Output directory: {output_dir}")
    print(f"Starting step: {start_step:,}")

    final_step = start_step
    stopped_early = False
    model.train()
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            for step in range(start_step, config.resolved_max_steps):
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                step_original_tokens = 0
                step_clone_tokens = 0

                for _ in range(config.gradient_accumulation_steps):
                    x, y, language_ids = train_stream.get_batch(
                        batch_size=config.micro_batch_size,
                        block_size=config.block_size,
                        device=device,
                        cloned_mapper=clone_mapper,
                        generator=generator,
                    )
                    with _autocast_context(device, precision):
                        _, loss = model(x, y)
                        if loss is None:
                            raise RuntimeError("model did not return training loss")
                        scaled_loss = loss / config.gradient_accumulation_steps
                    scaler.scale(scaled_loss).backward()
                    accumulated_loss += loss.item()

                    clone_sequences = int(language_ids.sum().item())
                    clone_tokens = clone_sequences * config.block_size
                    step_clone_tokens += clone_tokens
                    step_original_tokens += x.numel() - clone_tokens

                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip,
                )
                learning_rate = optimizer.param_groups[0]["lr"]
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                completed_step = step + 1
                final_step = completed_step
                original_tokens_seen += step_original_tokens
                clone_tokens_seen += step_clone_tokens
                tokens_seen = original_tokens_seen + clone_tokens_seen
                mean_train_loss = (
                    accumulated_loss / config.gradient_accumulation_steps
                )

                if completed_step % config.log_interval == 0 or completed_step == 1:
                    train_record = {
                        "type": "train",
                        "step": completed_step,
                        "train_loss": mean_train_loss,
                        "learning_rate": learning_rate,
                        "gradient_norm": float(gradient_norm.item()),
                        "tokens_seen": tokens_seen,
                        "original_tokens_seen": original_tokens_seen,
                        "clone_tokens_seen": clone_tokens_seen,
                    }
                    _write_log(log_file, train_record)
                    print(
                        f"step {completed_step:>6} | loss {mean_train_loss:.4f} | "
                        f"lr {learning_rate:.2e} | grad {gradient_norm.item():.3f} | "
                        f"tokens {tokens_seen:,}"
                    )

                should_evaluate = (
                    completed_step % config.eval_interval == 0
                    or completed_step == config.resolved_max_steps
                )
                if should_evaluate:
                    metrics = evaluate(
                        model,
                        validation_stream,
                        clone_mapper,
                        config,
                        device,
                        precision,
                    )
                    _write_log(
                        log_file,
                        {
                            "type": "validation",
                            "step": completed_step,
                            "learning_rate": learning_rate,
                            "tokens_seen": tokens_seen,
                            **metrics,
                        },
                    )
                    print(
                        f"validation | original {metrics['original_loss']:.4f} | "
                        f"clone {metrics['clone_loss']:.4f} | "
                        f"average {metrics['average_loss']:.4f}"
                    )

                    if metrics["average_loss"] < best_validation_loss:
                        best_validation_loss = metrics["average_loss"]
                        evaluations_without_improvement = 0
                        save_checkpoint(
                            output_dir / "best.pt",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            completed_step,
                            model_config,
                            config,
                            best_validation_loss,
                            tokens_seen,
                            original_tokens_seen,
                            clone_tokens_seen,
                            evaluations_without_improvement,
                            generator,
                        )
                    else:
                        evaluations_without_improvement += 1

                    if (
                        config.early_stopping_patience is not None
                        and evaluations_without_improvement
                        >= config.early_stopping_patience
                    ):
                        print("Early stopping patience reached.")
                        stopped_early = True
                        break

                if completed_step % config.checkpoint_interval == 0:
                    save_checkpoint(
                        output_dir / "latest.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        completed_step,
                        model_config,
                        config,
                        best_validation_loss,
                        tokens_seen,
                        original_tokens_seen,
                        clone_tokens_seen,
                        evaluations_without_improvement,
                        generator,
                    )
    except RuntimeError as error:
        if device.type == "cuda" and "out of memory" in str(error).lower():
            print(
                "CUDA out of memory. Set micro_batch_size: 4 in the YAML "
                "config and increase gradient_accumulation_steps to 8 if you "
                "want to keep an effective batch size of 32 sequences."
            )
        raise

    save_checkpoint(
        output_dir / "final.pt",
        model,
        optimizer,
        scheduler,
        scaler,
        final_step,
        model_config,
        config,
        best_validation_loss,
        tokens_seen,
        original_tokens_seen,
        clone_tokens_seen,
        evaluations_without_improvement,
        generator,
    )
    print(f"Training {'stopped early' if stopped_early else 'completed'}.")
    print(f"Actual tokens seen: {tokens_seen:,}")
    print(f"Original / clone: {original_tokens_seen:,} / {clone_tokens_seen:,}")
    print(f"Checkpoints and log: {output_dir}")


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError("YAML config must contain a mapping")

    flattened: dict[str, Any] = {}
    for key, value in document.items():
        if isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value
    valid_fields = {field.name for field in fields(TrainingConfig)}
    unknown = set(flattened) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    return flattened


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
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    args = vars(parser.parse_args())

    config_path = args.pop("config")
    values = _load_config_file(config_path) if config_path is not None else {}
    values.update({key: value for key, value in args.items() if value is not None})
    return TrainingConfig(**values)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
