#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-"${REPOSITORY_ROOT}/.venv"}

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${REPOSITORY_ROOT}/requirements.txt"
"${VENV_DIR}/bin/python" -m pip install --no-deps -e "${REPOSITORY_ROOT}"

"${VENV_DIR}/bin/python" -c \
  'import torch; print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")'

echo "Environment ready: ${VENV_DIR}"
