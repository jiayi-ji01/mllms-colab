# MLLMs: Original / Cloned-Language GPT

本项目从零训练 GPT-2 style decoder-only Transformer，比较 Original English 与
Cloned Language 的 Subject–Verb Agreement（SVA）行为，并通过 activation patching
研究两种 token space 是否使用相似的内部机制。

实验优先级是：

```text
Pretraining sanity
  → SVA sanity
  → causal component localization
  → original/clone circuit comparison
```

代码采用 `src/` layout。训练、linguistic evaluation、interpretability 和
visualization 相互分离；`mllms` 是唯一的用户命令入口。

## Repository structure

```text
.
├── configs/
│   ├── experiments/              # model + data + training + runtime
│   ├── evaluation/               # BLiMP / controlled SVA defaults
│   └── interpretability/         # activation-patching defaults
├── src/mllms/
│   ├── cli.py                    # unified command router
│   ├── config.py                 # validated YAML loading
│   ├── runtime.py                # device, precision, autocast
│   ├── model/
│   │   ├── config.py             # GPTConfig
│   │   ├── components.py         # attention, MLP, block, HookPoint
│   │   ├── transformer.py        # GPT, embeddings, final norm, LM head
│   │   └── loading.py            # inference checkpoint loading
│   ├── data/
│   │   ├── babylm.py
│   │   ├── tinystories.py
│   │   ├── cloned_language.py    # token-ID mapping only
│   │   └── token_stream.py       # memmap loading and batch construction
│   ├── tokenizer/
│   │   ├── sentencepiece.py
│   │   ├── train.py
│   │   └── tokenize.py
│   ├── training/
│   │   ├── config.py
│   │   ├── checkpoint.py
│   │   ├── engine.py
│   │   └── cli.py
│   ├── evaluation/
│   │   ├── language_model.py
│   │   ├── sanity.py
│   │   ├── blimp/
│   │   └── sva/
│   ├── interpretability/
│   │   └── activation_patching/
│   │       ├── metrics.py
│   │       ├── interventions.py
│   │       ├── results.py
│   │       └── runner.py
│   └── visualization/            # reads saved results; no model execution
├── scripts/                      # environment and cluster shell helpers
├── cluster/train.sbatch
├── tests/
├── book/                         # historical BabyLM result report
└── main.py                       # compatibility launcher
```

Generated data, tokenizer files, checkpoints and reports are excluded from Git.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
mllms --help
```

On a managed GPU cluster, install the PyTorch/CUDA build recommended by the
administrator before installing the remaining requirements. The helper performs the
same setup and prints CUDA availability:

```bash
bash scripts/setup_cluster_env.sh
```

## Configuration

Training experiments have one canonical YAML each:

- `configs/experiments/gpt12_babylm_clone.yaml`
- `configs/experiments/gpt12_tinystories_clone.yaml`

Each file contains the complete model, data, optimizer, schedule, seed and output
configuration for that experiment. Evaluation and patching defaults live in:

- `configs/evaluation/blimp.yaml`
- `configs/evaluation/sva.yaml`
- `configs/interpretability/activation_patching.yaml`

CLI flags override YAML values. Model/training parameters are not copied into the
evaluation configs.

## Data preparation

### BabyLM

```bash
mllms data prepare-babylm --output-dir data/babylm/raw
```

This downloads the pinned official cleaned BabyLM 100M train corpus and official
dev/test splits. A manifest records the resolved dataset revision and split counts.

### TinyStories

```bash
mllms data prepare --output-dir data/raw
```

The preparation step uses a fixed revision, seed and disjoint train/validation/test
selection.

## Tokenizer

BabyLM SentencePiece BPE:

```bash
mllms tokenizer train \
  --input data/babylm/raw/train.txt \
  --model-prefix artifacts/babylm_tokenizer/tokenizer \
  --vocab-size 16000 \
  --input-sentence-size 5000000

mllms data tokenize \
  --input-dir data/babylm/raw \
  --output-dir data/babylm/processed \
  --tokenizer artifacts/babylm_tokenizer/tokenizer.model \
  --no-train-token-limit
```

TinyStories:

```bash
mllms tokenizer train
mllms data tokenize --target-train-tokens 100000000
```

Raw preprocessing, tokenizer training, tokenization, dataset loading, cloned mapping
and batch construction are separate modules.

## Pretraining

BabyLM:

```bash
mllms train --config configs/experiments/gpt12_babylm_clone.yaml
```

TinyStories:

```bash
mllms train --config configs/experiments/gpt12_tinystories_clone.yaml
```

Resume explicitly:

```bash
mllms train \
  --config configs/experiments/gpt12_babylm_clone.yaml \
  --resume outputs/runs/gpt12_babylm_clone/latest.pt
```

The checkpoint contains model, optimizer, scheduler, scaler, counters, training
configuration and RNG states. Existing checkpoints remain loadable because model
attribute names, state-dict keys and checkpoint schema are unchanged.

Run pretraining sanity checks before linguistic evaluation:

```bash
mllms analyze sanity-check \
  --checkpoint outputs/runs/gpt12_babylm_clone/best.pt \
  --device cuda

mllms plot training --run-dir outputs/runs/gpt12_babylm_clone
```

## BLiMP SVA evaluation

```bash
RUN_DIR=outputs/runs/gpt12_babylm_clone
TOKENIZER=artifacts/babylm_tokenizer/tokenizer.model

mllms blimp download
mllms blimp prepare \
  --config configs/evaluation/blimp.yaml \
  --checkpoint "$RUN_DIR/best.pt" \
  --tokenizer "$TOKENIZER" \
  --output data/blimp/processed/babylm_agreement.jsonl
mllms blimp evaluate \
  --config configs/evaluation/blimp.yaml \
  --checkpoint "$RUN_DIR/best.pt" \
  --tokenizer "$TOKENIZER" \
  --data data/blimp/processed/babylm_agreement.jsonl \
  --scoring verb-logit \
  --device cuda \
  --output-dir "$RUN_DIR/blimp_sva_logit"
mllms plot blimp --results-dir "$RUN_DIR/blimp_sva_logit"
```

`verb-logit` only evaluates pairs with an identical prefix and one-token verb
alternatives. `conditional-logprob` remains available for the full compatible set.

## Controlled SVA evaluation

Build the project-owned Marvin--Linzen-style suite. Its dev/test vocabularies are
disjoint, answer verbs are single SentencePiece tokens, and every clean/corrupted
prompt differs only in the number-inflected form of one subject lemma:

```bash
mllms analyze build-controlled-sva \
  --config configs/evaluation/sva.yaml \
  --tokenizer "$TOKENIZER" \
  --output-dir data/sva/controlled_v1

mllms analyze evaluate-sva \
  --config configs/evaluation/sva.yaml \
  --checkpoint "$RUN_DIR/best.pt" \
  --tokenizer "$TOKENIZER" \
  --data data/sva/controlled_v1/test.jsonl \
  --device cuda \
  --output-dir "$RUN_DIR/controlled_sva"
```

The four controlled conditions are `simple`, `pp_attractor`, `object_relative`
and `subject_relative`. The clean prompt's matched and mismatched attractor
conditions are reported separately; changing subject number reverses this relation
in the corrupted prompt. `metadata.json` pins the tokenizer hash, seed, lexical
split and condition counts.

The CausalGym-derived suite remains available for comparison and causal-method
benchmarking:

```bash
mllms analyze prepare-sva \
  --config configs/evaluation/sva.yaml \
  --tokenizer "$TOKENIZER" \
  --output data/causalgym/sva_pairs.jsonl

mllms analyze evaluate-sva \
  --config configs/evaluation/sva.yaml \
  --checkpoint "$RUN_DIR/best.pt" \
  --tokenizer "$TOKENIZER" \
  --data data/causalgym/sva_pairs.jsonl \
  --device cuda \
  --output-dir "$RUN_DIR/causalgym_sva"

mllms plot sva --results-dir "$RUN_DIR/causalgym_sva"
```

The evaluator records accuracy, pair accuracy and correctly oriented logit
difference separately for original and clone. Only pairs passing the joint sanity
criterion are written to `sanity_pairs.jsonl`.

## Activation patching

Only proceed after inspecting SVA performance and the number/distribution of joint
sanity pairs.

```bash
mllms analyze activation-patching \
  --config configs/interpretability/activation_patching.yaml \
  --checkpoint "$RUN_DIR/best.pt" \
  --tokenizer "$TOKENIZER" \
  --data "$RUN_DIR/causalgym_sva/sanity_pairs.jsonl" \
  --language both \
  --device cuda \
  --output-dir "$RUN_DIR/causalgym_patching"
```

The runner patches these stable activation names:

```text
blocks.{layer}.resid_post
blocks.{layer}.attn_out
blocks.{layer}.mlp_out
blocks.{layer}.head_out
```

Metrics are saved as both raw `delta_ld` and normalized `recovery`. Plotting is a
separate result-only step:

```bash
mllms plot patching \
  --results-dir "$RUN_DIR/causalgym_patching" \
  --language original \
  --metric recovery
```

## Output layout

```text
outputs/runs/EXPERIMENT/
├── resolved_config.json
├── train_log.jsonl
├── latest.pt
├── best.pt
├── final.pt
├── sanity_check_lm/
├── blimp_sva_logit/
├── causalgym_sva/
└── causalgym_patching/
    ├── metadata.json
    ├── original/
    │   ├── per_example_scores.npz
    │   ├── mean_scores.csv
    │   └── *_mean.npy / *_count.npy
    └── clone/
```

Historical downloaded results remain under ignored `output/`; they are not used as
new defaults.

## Cluster execution

Submit the generic single-GPU Slurm template from the repository root:

```bash
sbatch --partition=YOUR_GPU_PARTITION cluster/train.sbatch
```

Override experiment and storage location without editing the script:

```bash
sbatch \
  --partition=YOUR_GPU_PARTITION \
  --export=ALL,CONFIG=configs/experiments/gpt12_tinystories_clone.yaml,RUN_DIR=/shared/USER/mllms/tinystories \
  cluster/train.sbatch
```

On a non-Slurm GPU server:

```bash
bash scripts/train_cluster.sh \
  configs/experiments/gpt12_babylm_clone.yaml \
  outputs/runs/gpt12_babylm_clone
```

The helper resumes `latest.pt` automatically and skips a completed `final.pt`.

## Tests

The test suite protects cloned mapping, checkpoint state-dict compatibility, SVA
metric direction and the core patching recovery invariant:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Important entry points

- Model configuration: `src/mllms/model/config.py`
- Transformer components and hooks: `src/mllms/model/components.py`
- GPT forward pass: `src/mllms/model/transformer.py`
- Cloned language mapping: `src/mllms/data/cloned_language.py`
- Training loop: `src/mllms/training/engine.py`
- SVA scoring: `src/mllms/evaluation/sva/scoring.py`
- Patching operations: `src/mllms/interpretability/activation_patching/interventions.py`
- Patching metrics: `src/mllms/interpretability/activation_patching/metrics.py`
- Visualization: `src/mllms/visualization/`
