"""Train or resume the cloned-language GPT."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import argparse
import json
import math
import random

import numpy as np
import torch

from checkpoint import TrainingState, load_checkpoint, save_checkpoint
from config import TrainingConfig, load_training_config
from model import GPT
from runtime import autocast_context, make_grad_scaler, select_device, select_precision
from token_data import ClonedMapper, TokenStream, load_tokenizer


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    max_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
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
            with autocast_context(device, precision):
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


def _write_log(log_file, record: dict[str, Any]) -> None:
    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    log_file.flush()


def train(config: TrainingConfig) -> None:
    """Run accumulated mixed-precision training and checkpointing."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = select_device(config.device)
    precision = select_precision(device)
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
    max_steps = config.resolve_max_steps(len(train_stream.tokens))
    warmup_steps = config.resolve_warmup_steps(max_steps)

    model_config = config.model_config()
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, config, max_steps, warmup_steps)
    scaler = make_grad_scaler(enabled=precision == "fp16")

    state = TrainingState()
    if config.resume is not None:
        state = load_checkpoint(
            Path(config.resume), model, optimizer, scheduler, scaler, generator, device,
        )

    def save(path: Path) -> None:
        save_checkpoint(
            path, model, optimizer, scheduler, scaler, state, config, generator,
            epoch=state.tokens_seen / len(train_stream.tokens),
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    log_path = output_dir / "train_log.jsonl"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    (output_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "model": asdict(model_config),
                "training": asdict(config),
                "resolved_max_steps": max_steps,
                "resolved_warmup_steps": warmup_steps,
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
    planned_tokens = max_steps * config.tokens_per_step
    print(f"Optimizer steps: {max_steps:,}")
    print(f"Planned tokens seen: {planned_tokens:,}")
    print(f"Nominal epochs: {planned_tokens / len(train_stream.tokens):.3f}")
    print(f"Warmup steps: {warmup_steps:,}")
    print(f"Output directory: {output_dir}")
    print(f"Starting step: {state.step:,}")

    stopped_early = False
    model.train()
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            for step in range(state.step, max_steps):
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
                    with autocast_context(device, precision):
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

                state.step = step + 1
                state.original_tokens_seen += step_original_tokens
                state.clone_tokens_seen += step_clone_tokens
                state.tokens_seen = state.original_tokens_seen + state.clone_tokens_seen
                mean_train_loss = (
                    accumulated_loss / config.gradient_accumulation_steps
                )

                if (
                    state.step % config.log_interval == 0
                    or state.step == 1
                    or state.step == max_steps
                ):
                    train_record = {
                        "type": "train",
                        "step": state.step,
                        "train_loss": mean_train_loss,
                        "learning_rate": learning_rate,
                        "gradient_norm": float(gradient_norm.item()),
                        "tokens_seen": state.tokens_seen,
                        "original_tokens_seen": state.original_tokens_seen,
                        "clone_tokens_seen": state.clone_tokens_seen,
                    }
                    _write_log(log_file, train_record)
                    print(
                        f"step {state.step:>6} | loss {mean_train_loss:.4f} | "
                        f"lr {learning_rate:.2e} | grad {gradient_norm.item():.3f} | "
                        f"tokens {state.tokens_seen:,}"
                    )

                should_evaluate = (
                    state.step % config.eval_interval == 0
                    or state.step == max_steps
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
                            "step": state.step,
                            "learning_rate": learning_rate,
                            "tokens_seen": state.tokens_seen,
                            **metrics,
                        },
                    )
                    print(
                        f"validation | original {metrics['original_loss']:.4f} | "
                        f"clone {metrics['clone_loss']:.4f} | "
                        f"average {metrics['average_loss']:.4f}"
                    )
                    state.validation_loss = metrics["average_loss"]

                    if state.validation_loss < state.best_validation_loss:
                        state.best_validation_loss = state.validation_loss
                        state.evaluations_without_improvement = 0
                        save(output_dir / "best.pt")
                    else:
                        state.evaluations_without_improvement += 1

                    if (
                        config.early_stopping_patience is not None
                        and state.evaluations_without_improvement
                        >= config.early_stopping_patience
                    ):
                        print("Early stopping patience reached.")
                        stopped_early = True
                        break

                if state.step % config.checkpoint_interval == 0:
                    checkpoint_path = (
                        checkpoint_dir / f"step_{state.step:06d}.pt"
                    )
                    save(checkpoint_path)
                    print(f"Checkpoint: {checkpoint_path}")
    except RuntimeError as error:
        if device.type == "cuda" and "out of memory" in str(error).lower():
            print(
                "CUDA out of memory. Set micro_batch_size: 4 in the YAML "
                "config and increase gradient_accumulation_steps to 8 if you "
                "want to keep an effective batch size of 32 sequences."
            )
        raise

    save(output_dir / "last.pt")
    print(f"Training {'stopped early' if stopped_early else 'completed'}.")
    print(f"Actual tokens seen: {state.tokens_seen:,}")
    print(f"Original / clone: {state.original_tokens_seen:,} / {state.clone_tokens_seen:,}")
    print(f"Checkpoints and log: {output_dir}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
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
    config = load_training_config(config_path)
    overrides = {key: value for key, value in values.items() if value is not None}
    if not overrides:
        return config
    merged = {**vars(config), **overrides}
    return TrainingConfig(**merged)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
