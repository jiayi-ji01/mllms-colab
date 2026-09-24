"""Train the GPT model on balanced original and cloned token streams."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from mllms.data.cloned_language import ClonedMapper
from mllms.data.token_stream import TokenStream
from mllms.model.config import GPTConfig
from mllms.model.transformer import GPT
from mllms.runtime import (
    autocast_context,
    make_grad_scaler,
    select_device,
    select_precision,
)
from mllms.tokenizer.sentencepiece import load_tokenizer
from mllms.training.checkpoint import load_checkpoint, save_checkpoint
from mllms.training.config import TrainingConfig


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
    scheduler = _make_scheduler(optimizer, config, max_steps, warmup_steps)
    scaler = make_grad_scaler(enabled=precision == "fp16")

    start_step = 0
    best_validation_loss = float("inf")
    last_validation_loss = float("inf")
    tokens_seen = 0
    original_tokens_seen = 0
    clone_tokens_seen = 0
    evaluations_without_improvement = 0
    if config.resume is not None:
        (
            start_step,
            best_validation_loss,
            last_validation_loss,
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
    print(f"Starting step: {start_step:,}")

    final_step = start_step
    stopped_early = False
    model.train()
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            for step in range(start_step, max_steps):
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

                completed_step = step + 1
                final_step = completed_step
                original_tokens_seen += step_original_tokens
                clone_tokens_seen += step_clone_tokens
                tokens_seen = original_tokens_seen + clone_tokens_seen
                mean_train_loss = (
                    accumulated_loss / config.gradient_accumulation_steps
                )

                if (
                    completed_step % config.log_interval == 0
                    or completed_step == 1
                    or completed_step == max_steps
                ):
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
                    or completed_step == max_steps
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
                    last_validation_loss = metrics["average_loss"]

                    if last_validation_loss < best_validation_loss:
                        best_validation_loss = last_validation_loss
                        evaluations_without_improvement = 0
                        save_checkpoint(
                            output_dir / "best.pt",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            completed_step,
                            tokens_seen / len(train_stream.tokens),
                            model_config,
                            config,
                            best_validation_loss,
                            last_validation_loss,
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
                    checkpoint_path = (
                        checkpoint_dir / f"step_{completed_step:06d}.pt"
                    )
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        completed_step,
                        tokens_seen / len(train_stream.tokens),
                        model_config,
                        config,
                        best_validation_loss,
                        last_validation_loss,
                        tokens_seen,
                        original_tokens_seen,
                        clone_tokens_seen,
                        evaluations_without_improvement,
                        generator,
                    )
                    print(f"Checkpoint: {checkpoint_path}")
    except RuntimeError as error:
        if device.type == "cuda" and "out of memory" in str(error).lower():
            print(
                "CUDA out of memory. Set micro_batch_size: 4 in the YAML "
                "config and increase gradient_accumulation_steps to 8 if you "
                "want to keep an effective batch size of 32 sequences."
            )
        raise

    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        scheduler,
        scaler,
        final_step,
        tokens_seen / len(train_stream.tokens),
        model_config,
        config,
        best_validation_loss,
        last_validation_loss,
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
