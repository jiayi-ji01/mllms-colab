#!/usr/bin/env bash
#SBATCH --job-name=mllms-v2
#SBATCH --partition=students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

REPOSITORY_ROOT=${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}
cd "${REPOSITORY_ROOT}"
mkdir -p logs
nvidia-smi
srun scripts/run_v2_analysis.sh
