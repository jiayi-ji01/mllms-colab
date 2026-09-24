#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --job-name=mllms-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

REPOSITORY_ROOT=${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}
cd "${REPOSITORY_ROOT}"

export HF_HOME=${HF_HOME:-"${REPOSITORY_ROOT}/.cache/huggingface"}
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

srun .venv/bin/mllms data prepare-wikipedia \
  --output-dir data/wikipedia/raw \
  --revision e6057dc557255a03c9c3c47ceab0eb44353b1bc5

srun .venv/bin/mllms tokenizer train \
  --input data/wikipedia/raw/train.txt \
  --model-prefix artifacts/wikipedia_tokenizer/tokenizer \
  --vocab-size 16000 \
  --input-sentence-size 5000000

srun .venv/bin/mllms data tokenize \
  --input-dir data/wikipedia/raw \
  --output-dir data/wikipedia/processed \
  --tokenizer artifacts/wikipedia_tokenizer/tokenizer.model \
  --no-train-token-limit
