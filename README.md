# Cross-language SVA circuits without shared token IDs

This repository trains a 12-layer decoder-only Transformer on English Wikipedia
and a cloned language with a disjoint token-ID partition. It then tests whether
subject-number information transfers causally between the two token spaces.

The primary result is a bidirectional, asymmetric transfer effect at attention
head L8H3. The full Chinese report is in
[`reports/cross_language_sva/report_zh.md`](reports/cross_language_sva/report_zh.md),
and the plotting notebook is
[`notebooks/cross_language_dashboard.ipynb`](notebooks/cross_language_dashboard.ipynb).

```text
Wikipedia preparation
  → original/clone language-model training
  → controlled SVA v2 evaluation
  → four-direction activation patching
  → paired trajectory statistics and report figures
```

## Repository contents

```text
configs/                         Current training, SVA and patching configurations
data/sva/controlled_v2/          Versioned dev/test evaluation pairs and metadata
src/mllms/                       Training, evaluation and interpretability package
scripts/                         Cluster setup, execution and report construction
cluster/                         Slurm entry points for the recorded experiment
tests/                           Unit and interface tests
notebooks/cross_language_dashboard.ipynb
reports/cross_language_sva/      Report, compact evidence tables and PNG figures
docs/experiment_log.md           Public experiment decisions and limitations
```

Generated corpora, tokenizer files, checkpoints, per-example activation arrays,
logs and local caches are intentionally excluded from Git.

## Environment

Python 3.10 or newer is required. The setup script installs the pinned cluster
dependencies and this package in editable mode:

```bash
bash scripts/setup_cluster_env.sh
source .venv/bin/activate
mllms --help
```

For notebook execution, install the reporting extra:

```bash
python -m pip install -e '.[report]'
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
`reports/cross_language_sva/data/manifest.json` before plotting.

### 2. Rebuild the compact statistics

This requires locally generated checkpoints, tokenizer, SVA evaluations and raw
patching outputs at the paths recorded in the configs:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/build_report.py
```

The builder validates checkpoint, tokenizer and dataset hashes; verifies the
fixed 1,280-pair cohort at every checkpoint; and rejects missing, duplicate or
reordered sample IDs.

### 3. Re-run the complete experiment

Prepare the pinned Wikipedia snapshot and tokenizer:

```bash
mllms data prepare-wikipedia \
  --output-dir data/wikipedia/raw \
  --revision e6057dc557255a03c9c3c47ceab0eb44353b1bc5

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

Train and build the controlled evaluation data:

```bash
mllms train --config configs/experiments/gpt12_wikipedia_clone.yaml

mllms analyze build-controlled-sva \
  --config configs/evaluation/sva_wikipedia_v2.yaml
```

Evaluate a checkpoint and run the confirmatory patching configuration:

```bash
CHECKPOINT=outputs/runs/gpt12_wikipedia_clone/best.pt

mllms analyze evaluate-sva \
  --config configs/evaluation/sva_wikipedia_v2.yaml \
  --checkpoint "$CHECKPOINT" \
  --output-dir outputs/evaluation/wikipedia_sva_v2/best \
  --device cuda

mllms analyze activation-patching \
  --config configs/interpretability/activation_patching_wikipedia_v2_confirm.yaml \
  --checkpoint "$CHECKPOINT" \
  --output-dir outputs/interpretability/wikipedia_cross_language_v2_confirm \
  --device cuda
```

The Slurm entry points reproduce the recorded large runs:

- `cluster/prepare-wikipedia.sbatch`
- `cluster/train.sbatch`
- `cluster/sva-v2-array.sbatch`
- `cluster/patch-trajectory-v2-array.sbatch`
- `cluster/analyze-v2.sbatch`, parameterized through `STAGE`

Large artifacts are not distributed with this repository. Reproducing the
reported numerical results from scratch therefore requires retraining the model.

## Main command interface

```text
mllms data prepare-wikipedia
mllms data tokenize
mllms tokenizer train
mllms train --config CONFIG
mllms analyze build-controlled-sva
mllms analyze evaluate-sva
mllms analyze sanity-check
mllms analyze activation-patching
mllms analyze patching-statistics
mllms plot training|sva|patching
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The tests cover cloned-token mapping, training/checkpoint invariants, controlled
SVA generation and scoring, source-to-target patching, controls, paired
statistics, configuration defaults and the public command interface.
