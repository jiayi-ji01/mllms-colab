#!/usr/bin/env bash
#SBATCH --job-name=mllms-patch-trajectory-v2
#SBATCH --partition=students
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

REPOSITORY_ROOT=${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}
CHECKPOINTS=(
  outputs/runs/gpt12_wikipedia_clone/checkpoints/step_005000.pt
  outputs/runs/gpt12_wikipedia_clone/checkpoints/step_015000.pt
  outputs/runs/gpt12_wikipedia_clone/checkpoints/step_030000.pt
  outputs/runs/gpt12_wikipedia_clone/checkpoints/step_050000.pt
  outputs/runs/gpt12_wikipedia_clone/checkpoints/step_065000.pt
  outputs/runs/gpt12_wikipedia_clone/best.pt
)
CHECKPOINT=${CHECKPOINTS[${SLURM_ARRAY_TASK_ID}]}
STEP_NAME=$(basename "${CHECKPOINT}" .pt)

cd "${REPOSITORY_ROOT}"
mkdir -p logs
export STAGE=patch
export CHECKPOINT
export PATCH_CONFIG=configs/interpretability/activation_patching_wikipedia_v2_trajectory.yaml
export OUTPUT_DIR="outputs/interpretability/wikipedia_cross_language_v2_trajectory/${STEP_NAME}"
nvidia-smi
srun scripts/run_v2_analysis.sh
