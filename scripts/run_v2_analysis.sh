#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR=${VENV_DIR:-"${REPOSITORY_ROOT}/.venv"}
STAGE=${STAGE:?Set STAGE to sva, patch, test-ppl, or statistics}

cd "${REPOSITORY_ROOT}"
if [[ ! -x "${VENV_DIR}/bin/mllms" ]]; then
  echo "Missing environment: ${VENV_DIR}. Run scripts/setup_cluster_env.sh first." >&2
  exit 1
fi

case "${STAGE}" in
  sva)
    CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT for SVA evaluation}
    OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR for SVA evaluation}
    exec "${VENV_DIR}/bin/mllms" analyze evaluate-sva \
      --config configs/evaluation/sva_wikipedia_v2.yaml \
      --checkpoint "${CHECKPOINT}" \
      --output-dir "${OUTPUT_DIR}" \
      --device cuda
    ;;
  patch)
    CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT for activation patching}
    OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR for activation patching}
    PATCH_CONFIG=${PATCH_CONFIG:-configs/interpretability/activation_patching_wikipedia_v2_confirm.yaml}
    exec "${VENV_DIR}/bin/mllms" analyze activation-patching \
      --config "${PATCH_CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --output-dir "${OUTPUT_DIR}" \
      --device cuda
    ;;
  test-ppl)
    CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT for held-out test evaluation}
    OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR for held-out test evaluation}
    exec "${VENV_DIR}/bin/mllms" analyze sanity-check \
      --config configs/experiments/gpt12_wikipedia_clone.yaml \
      --checkpoint "${CHECKPOINT}" \
      --output-dir "${OUTPUT_DIR}" \
      --split test \
      --full-validation \
      --device cuda
    ;;
  statistics)
    RESULTS_DIR=${RESULTS_DIR:?Set RESULTS_DIR for patching statistics}
    OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR for patching statistics}
    exec "${VENV_DIR}/bin/mllms" analyze patching-statistics \
      --results-dir "${RESULTS_DIR}" \
      --data data/sva/controlled_v2/test.jsonl \
      --output-dir "${OUTPUT_DIR}" \
      --iterations 10000 \
      --seed 42
    ;;
  *)
    echo "Unknown STAGE: ${STAGE}" >&2
    exit 2
    ;;
esac
