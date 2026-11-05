#!/bin/bash
# One-time setup script on UPPMAX Pelle.
#
# Creates separate inference (.venv-infer) and training (.venv-train) virtual
# environments, installs pinned dependencies and the editable `lrmt` package,
# downloads & verifies FLORES+, and optionally pre-caches the 8B model weights.
#
# Usage on Pelle (recommend running inside a short interactive session or login node):
#   module load python/3.12   # or python/3.11 if 3.12 is unavailable
#   hf auth login             # once, if not already logged in
#   bash slurm/setup_pelle.sh

set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "Using Python interpreter: $(command -v "${PYTHON_BIN}") ($(${PYTHON_BIN} --version))"

echo "[1/4] Creating and populating inference venv (${VENV_INFER})..."
if [[ ! -d "${VENV_INFER}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_INFER}"
fi
activate_infer
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-infer.txt -r requirements-analysis.txt
python -m pip install -e .
deactivate

echo "[2/4] Creating and populating training venv (${VENV_TRAIN})..."
if [[ ! -d "${VENV_TRAIN}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_TRAIN}"
fi
activate_train
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-train.txt -r requirements-analysis.txt
python -m pip install -e .

echo "[3/4] Downloading and validating FLORES+ parallel data..."
python scripts/download_flores.py

echo "[4/4] Verifying nested data pools and tokenizer parity..."
python scripts/make_pools.py
python scripts/tokenizer_parity.py

deactivate
echo "Setup complete! Virtual environments are ready:"
echo "  Inference : ${VENV_INFER}"
echo "  Training  : ${VENV_TRAIN}"
echo "  HF Cache  : ${HF_HOME}"
