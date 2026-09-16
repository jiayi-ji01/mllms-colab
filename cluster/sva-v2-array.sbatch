#!/usr/bin/env bash
#SBATCH --job-name=mllms-sva-v2
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
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
  outputs/runs/gpt12_wikipedia_clone/checkpoints/step_075000.pt
  outputs/runs/gpt12_wikipedia_clone/best.pt
)
CHECKPOINT=${CHECKPOINTS[${SLURM_ARRAY_TASK_ID}]}
STEP_NAME=$(basename "${CHECKPOINT}" .pt)

cd "${REPOSITORY_ROOT}"
mkdir -p logs
export STAGE=sva
export CHECKPOINT
export OUTPUT_DIR="outputs/evaluation/wikipedia_sva_v2/${STEP_NAME}"
nvidia-smi
srun scripts/run_v2_analysis.sh
