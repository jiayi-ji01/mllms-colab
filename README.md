# MLLMs: Wikipedia + Cloned-Language GPT

This repository trains one 12-layer decoder-only Transformer on English Wikipedia
in balanced original/clone token spaces, evaluates subject–verb agreement (SVA) on
the fixed project-owned controlled suite, and prepares sanity pairs for later
activation patching.

```text
English Wikipedia pretraining
  → controlled SVA evaluation at multiple checkpoints
  → joint original/clone sanity-pair selection
  → activation patching
```

The controlled SVA suite is evaluation-only and is never mixed into pretraining.
TinyStories and BabyLM implementations/configs remain as historical references, but
the active experiment is `gpt12_wikipedia_clone`.

## Repository structure

```text
configs/
  experiments/gpt12_wikipedia_clone.yaml
  evaluation/sva_wikipedia.yaml
  interpretability/activation_patching_wikipedia.yaml
data/sva/controlled_v1/              # versioned canonical controlled benchmark
src/mllms/
  data/wikipedia.py                  # download, deterministic article split
  data/cloned_language.py            # original/clone ID mapping
  tokenizer/                         # SentencePiece training and token streams
  model/                             # unchanged GPT model and hook points
  training/                          # training, checkpointing, exact resume
  evaluation/sva/                    # controlled evaluation and sanity selection
  interpretability/activation_patching/
  visualization/                    # result-only plots
scripts/setup_cluster_env.sh
scripts/train_cluster.sh
cluster/train.sbatch
tests/
```

Generated Wikipedia text/token streams, tokenizer models, outputs, logs and model
checkpoints are ignored by Git. `data/sva/controlled_v1/` is intentionally tracked.
Its canonical prompts are re-tokenized by the evaluator with the tokenizer supplied
on the command line, so its older stored token IDs cannot silently contaminate a
Wikipedia-tokenizer evaluation.

## Environment

Python 3.10 or newer is required.

```bash
bash scripts/setup_cluster_env.sh
source .venv/bin/activate
mllms --help
```

On a managed cluster, load the site-recommended Python/CUDA module first if needed.
The setup script creates `.venv`, installs `requirements.txt`, installs this project
editable, and reports whether PyTorch can see CUDA.

## Wikipedia data and tokenizer

The preparation command streams the pinned `wikimedia/wikipedia` English
`20231101.en` snapshot. It writes whole, hash-disjoint articles until it reaches
100M train words, 1M validation words and 1M test words. A manifest records the
resolved dataset revision, license, seed and split statistics.

```bash
mllms data prepare-wikipedia --output-dir data/wikipedia/raw

mllms tokenizer train \
  --input data/wikipedia/raw/train.txt \
  --model-prefix artifacts/wikipedia_tokenizer/tokenizer \
  --vocab-size 16000 \
  --input-sentence-size 5000000

mllms data tokenize \
  --input-dir data/wikipedia/raw \
  --output-dir data/wikipedia/processed \
  --tokenizer artifacts/wikipedia_tokenizer/tokenizer.model \
  --no-train-token-limit
```

The SentencePiece BPE settings, vocabulary size, original/clone mapping and batch
construction are unchanged from the BabyLM baseline.

## Training and resume

```bash
mllms train --config configs/experiments/gpt12_wikipedia_clone.yaml
```

The model, optimizer, batch and context parameters match
`gpt12_babylm_clone`. The Wikipedia run uses four nominal epochs: with the balanced
`p_clone: 0.5` mixture, the original and clone token spaces each receive about two
dataset epochs. Dataset/tokenizer/output paths and the archival interval are also
experiment-specific. Checkpoints are written atomically as:

```text
outputs/runs/gpt12_wikipedia_clone/
  resolved_config.json
  train_log.jsonl
  checkpoints/step_005000.pt
  checkpoints/step_010000.pt
  ...
  best.pt
  last.pt
```

Each checkpoint includes model, optimizer, scheduler and scaler state; global step;
nominal epoch; latest/best validation loss; token counters; complete config; and
Python, NumPy, PyTorch and sampling-generator RNG states.

```bash
mllms train \
  --config configs/experiments/gpt12_wikipedia_clone.yaml \
  --resume outputs/runs/gpt12_wikipedia_clone/checkpoints/step_010000.pt
```

`scripts/train_cluster.sh` automatically resumes the newest numbered checkpoint and
skips a run that already has `last.pt`.

## Controlled SVA

Validate that all fixed prompts can be encoded by the Wikipedia tokenizer:

```bash
python -c 'from pathlib import Path; from mllms.tokenizer.sentencepiece import load_tokenizer; from mllms.evaluation.sva.pairs import read_pairs; t=load_tokenizer(Path("artifacts/wikipedia_tokenizer/tokenizer.model")); p=read_pairs(Path("data/sva/controlled_v1/test.jsonl"), tokenizer=t); print(f"validated {len(p)} controlled SVA pairs")'
```

Evaluate any final or intermediate checkpoint:

```bash
CHECKPOINT=outputs/runs/gpt12_wikipedia_clone/last.pt
STEP_NAME=$(basename "${CHECKPOINT}" .pt)
mllms analyze evaluate-sva \
  --config configs/evaluation/sva_wikipedia.yaml \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "outputs/evaluation/wikipedia_sva/${STEP_NAME}" \
  --device cuda
```

The evaluator reports accuracy, pair accuracy and oriented answer-sequence
log-probability difference for the original and clone languages across `simple`,
`pp_attractor`, `object_relative` and `subject_relative`. This reduces exactly to
the old logit difference for single-token answers and also supports multi-token
answers. Pairs that pass the joint clean/corrupted criterion in both languages are
written to `sanity_pairs.jsonl`.

```bash
mllms plot sva \
  --results-dir "outputs/evaluation/wikipedia_sva/${STEP_NAME}"
```

## Activation-patching preparation

Use the checkpoint-specific sanity set. Patching is deliberately not part of the
training loop.

```bash
mllms analyze activation-patching \
  --config configs/interpretability/activation_patching_wikipedia.yaml \
  --checkpoint "${CHECKPOINT}" \
  --data "outputs/evaluation/wikipedia_sva/${STEP_NAME}/sanity_pairs.jsonl" \
  --output-dir "outputs/interpretability/wikipedia_patching/${STEP_NAME}" \
  --device cuda
```

Stable hook locations remain visible as `blocks.{layer}.resid_post`,
`blocks.{layer}.attn_out`, `blocks.{layer}.mlp_out` and
`blocks.{layer}.head_out`. Plotting only reads saved results:

```bash
mllms plot patching \
  --results-dir "outputs/interpretability/wikipedia_patching/${STEP_NAME}" \
  --language original \
  --metric recovery
```

## Cluster execution

After preparing the environment and data, submit from the repository root:

```bash
mkdir -p logs
sbatch --partition=YOUR_GPU_PARTITION cluster/train.sbatch
```

The template intentionally leaves the institution-specific partition/account to the
submission command. Override storage or config without editing it:

```bash
mkdir -p logs
sbatch \
  --partition=YOUR_GPU_PARTITION \
  --export=ALL,CONFIG=configs/experiments/gpt12_wikipedia_clone.yaml,RUN_DIR=/shared/USER/mllms/gpt12_wikipedia_clone \
  cluster/train.sbatch
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Main entry points

- Wikipedia preparation: `src/mllms/data/wikipedia.py`
- Tokenizer/tokenization: `src/mllms/tokenizer/train.py`, `tokenize.py`
- Model and hook points: `src/mllms/model/transformer.py`, `components.py`
- Training: `src/mllms/training/engine.py`
- Checkpoint/resume: `src/mllms/training/checkpoint.py`
- Controlled SVA: `src/mllms/evaluation/sva/controlled.py`, `evaluate.py`
- Patching: `src/mllms/interpretability/activation_patching/runner.py`
- Visualization: `src/mllms/visualization/`
