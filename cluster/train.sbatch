#!/usr/bin/env bash
#SBATCH --job-name=mllms-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

REPOSITORY_ROOT=${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}
CONFIG=${CONFIG:-configs/experiments/gpt12_wikipedia_clone.yaml}
RUN_DIR=${RUN_DIR:-outputs/runs/gpt12_wikipedia_clone}

cd "${REPOSITORY_ROOT}"
nvidia-smi
srun scripts/train_cluster.sh "${CONFIG}" "${RUN_DIR}"
