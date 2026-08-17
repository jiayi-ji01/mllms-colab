#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:-configs/experiments/gpt12_babylm_clone.yaml}
RUN_DIR=${2:-outputs/runs/gpt12_babylm_clone}
VENV_DIR=${VENV_DIR:-"${REPOSITORY_ROOT}/.venv"}

cd "${REPOSITORY_ROOT}"

if [[ ! -x "${VENV_DIR}/bin/mllms" ]]; then
  echo "Missing environment: ${VENV_DIR}. Run scripts/setup_cluster_env.sh first." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
if [[ -f "${RUN_DIR}/final.pt" ]]; then
  echo "Training already completed: ${RUN_DIR}/final.pt"
  exit 0
fi

TRAIN_ARGS=(
  train
  --config "${CONFIG}"
  --output-dir "${RUN_DIR}"
  --device cuda
)
if [[ -f "${RUN_DIR}/latest.pt" ]]; then
  TRAIN_ARGS+=(--resume "${RUN_DIR}/latest.pt")
  echo "Resuming from ${RUN_DIR}/latest.pt"
fi

exec "${VENV_DIR}/bin/mllms" "${TRAIN_ARGS[@]}"
