# Cross-language SVA information transfer

This repository trains a 12-layer decoder-only Transformer on English Wikipedia
and a cloned language with a disjoint token-ID partition. It then tests whether
subject-number information transfers causally between the two token spaces.

The primary result is a bidirectional, asymmetric transfer effect at attention
head L8H3. The [compact evidence](evidence/cross_language_sva/README.md) and
[plotting notebook](notebooks/cross_language_dashboard.ipynb) support inspection
and redrawing of the reported results.

```text
Wikipedia preparation
  → original/clone language-model training
  → controlled SVA v2 evaluation
  → four-direction activation patching
  → paired statistical analysis
```

## Repository contents

```text
configs/                         Current training, SVA and patching configurations
data/sva/controlled_v2/          Versioned dev/test evaluation pairs and metadata
scripts/                         Experiment and analysis entry points
tests/                           Unit and interface tests
notebooks/cross_language_dashboard.ipynb
evidence/cross_language_sva/     Compact result tables and provenance
docs/experiment_log.md           Public experiment decisions and limitations
```

Reports, paper drafts, cluster submission files, generated corpora, tokenizer
files, checkpoints, per-example activation arrays, logs and caches stay local.

## Environment

Python 3.10 or newer is required. Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/train.py --help
```

For notebook execution, install these additional dependencies:

```bash
python -m pip install "matplotlib==3.11.1" "nbformat>=5,<6" "nbconvert>=7,<8" "ipykernel>=6,<7"
```

## Reproduction levels

### 1. Inspect the reported evidence

The tracked compact CSV/JSON files are sufficient to inspect every reported
number and redraw all ten figures. They do not require model checkpoints or raw
activation arrays.

```bash
python -m nbconvert \
  --to notebook \
  --execute notebooks/cross_language_dashboard.ipynb \
  --output /tmp/cross_language_dashboard.executed.ipynb
```

The notebook verifies the compact-file hashes recorded in
[`evidence/cross_language_sva/data/manifest.json`](evidence/cross_language_sva/data/manifest.json)
before plotting. It writes PNG files to an ignored local reports directory.

### 2. Re-run the experiment through statistical analysis

The compact evidence and plotting notebook are retained as recorded results.
The notebook redraws figures from those tables. Use the commands below to
recreate numerical results with the current scripts. The old `mllms` command
and Python package are retired.

Prepare the pinned Wikipedia snapshot and tokenizer:

```bash
python scripts/prepare_wikipedia.py \
  --output-dir data/wikipedia/raw \
  --revision e6057dc557255a03c9c3c47ceab0eb44353b1bc5

python scripts/train_tokenizer.py \
  --input data/wikipedia/raw/train.txt \
  --model-prefix artifacts/wikipedia_tokenizer/tokenizer \
  --vocab-size 16000 \
  --input-sentence-size 5000000

python scripts/tokenize_data.py \
  --input-dir data/wikipedia/raw \
  --output-dir data/wikipedia/processed \
  --tokenizer artifacts/wikipedia_tokenizer/tokenizer.model \
  --no-train-token-limit
```

Train and build the controlled evaluation data:

```bash
python scripts/train.py --config configs/experiments/gpt12_wikipedia_clone.yaml

python scripts/build_sva.py \
  --config configs/evaluation/sva_wikipedia_v2.yaml
```

Evaluate a checkpoint and run the confirmatory patching configuration:

```bash
CHECKPOINT=outputs/runs/gpt12_wikipedia_clone/best.pt

python scripts/evaluate_sva.py \
  --config configs/evaluation/sva_wikipedia_v2.yaml \
  --checkpoint "$CHECKPOINT" \
  --output-dir outputs/evaluation/wikipedia_sva_v2/best \
  --device cuda

python scripts/run_patching.py \
  --config configs/interpretability/activation_patching_wikipedia_v2_confirm.yaml \
  --checkpoint "$CHECKPOINT" \
  --output-dir outputs/interpretability/wikipedia_cross_language_v2_confirm \
  --device cuda
```

Evaluate held-out perplexity and compare patching effects with controls:

```bash
python scripts/evaluate_lm.py \
  --checkpoint "$CHECKPOINT" \
  --config configs/experiments/gpt12_wikipedia_clone.yaml \
  --split test --full-validation --device cuda

python scripts/patching_statistics.py \
  --results-dir outputs/interpretability/wikipedia_cross_language_v2_confirm \
  --data data/sva/controlled_v2/test.jsonl \
  --output-dir outputs/interpretability/wikipedia_cross_language_v2_statistics \
  --iterations 10000 --seed 42
```

The loss/perplexity CSV fields and `sanity_check_summary.json` filename are retained.
Prediction examples and random-model comparisons are no longer produced.

Large artifacts are not distributed with this repository. Reproducing the
reported numerical results from scratch therefore requires retraining the model.

## Main command interface

```text
python scripts/prepare_wikipedia.py
python scripts/tokenize_data.py
python scripts/train_tokenizer.py
python scripts/train.py --config CONFIG
python scripts/build_sva.py
python scripts/evaluate_sva.py
python scripts/evaluate_lm.py
python scripts/run_patching.py
python scripts/patching_statistics.py
```

## Tests

```bash
PYTHONPATH=scripts python -m unittest discover -s tests -v
```

The tests cover cloned-token mapping, training/checkpoint invariants, controlled
SVA generation and scoring, source-to-target patching, controls, paired
statistics, configuration defaults and the public command interface.
