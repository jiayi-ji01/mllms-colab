#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:-configs/experiments/gpt12_wikipedia_clone.yaml}
RUN_DIR=${2:-outputs/runs/gpt12_wikipedia_clone}
VENV_DIR=${VENV_DIR:-"${REPOSITORY_ROOT}/.venv"}

cd "${REPOSITORY_ROOT}"

if [[ ! -x "${VENV_DIR}/bin/mllms" ]]; then
  echo "Missing environment: ${VENV_DIR}. Run scripts/setup_cluster_env.sh first." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
if [[ -f "${RUN_DIR}/last.pt" ]]; then
  echo "Training already completed: ${RUN_DIR}/last.pt"
  exit 0
fi

TRAIN_ARGS=(
  train
  --config "${CONFIG}"
  --output-dir "${RUN_DIR}"
  --device cuda
)
shopt -s nullglob
CHECKPOINTS=("${RUN_DIR}"/checkpoints/step_*.pt)
if (( ${#CHECKPOINTS[@]} > 0 )); then
  LATEST_CHECKPOINT=${CHECKPOINTS[${#CHECKPOINTS[@]}-1]}
  TRAIN_ARGS+=(--resume "${LATEST_CHECKPOINT}")
  echo "Resuming from ${LATEST_CHECKPOINT}"
fi

exec "${VENV_DIR}/bin/mllms" "${TRAIN_ARGS[@]}"
