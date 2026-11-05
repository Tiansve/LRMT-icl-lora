#!/bin/bash
# Shared environment configuration sourced by all Slurm batch scripts on UPPMAX Pelle.
# Do not submit this file directly with sbatch; it is sourced by slurm/0*.sh.

set -euo pipefail

# Resolve repository root regardless of where sbatch was invoked from.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${PROJECT_ROOT}"
mkdir -p slurm/logs

# Keep Hugging Face cache inside the project storage directory (not /home/$USER)
# so 8B model weights do not exhaust the small home directory quota on UPPMAX.
export HF_HOME="${LRMT_HF_HOME:-${PROJECT_ROOT}/.hf_cache}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Virtual environment locations (created once by slurm/setup_pelle.sh).
export VENV_INFER="${LRMT_VENV_INFER:-${PROJECT_ROOT}/.venv-infer}"
export VENV_TRAIN="${LRMT_VENV_TRAIN:-${PROJECT_ROOT}/.venv-train}"

# Ordered 12 source languages for Slurm job arrays (indices 0..11):
# - Indices 0..3:  Pilot batch (4 languages spanning high to extreme low resource)
# - Indices 4..11: Remaining 8 source languages
export SOURCE_LANGS=(
    swh_Latn
    kaz_Cyrl
    amh_Ethi
    sat_Olck
    hau_Latn
    yor_Latn
    quy_Latn
    ayr_Latn
    khk_Cyrl
    uig_Arab
    tir_Ethi
    dzo_Tibt
)

activate_infer() {
    if [[ ! -f "${VENV_INFER}/bin/activate" ]]; then
        echo "ERROR: Inference venv not found at ${VENV_INFER}. Run bash slurm/setup_pelle.sh first." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${VENV_INFER}/bin/activate"
}

activate_train() {
    if [[ ! -f "${VENV_TRAIN}/bin/activate" ]]; then
        echo "ERROR: Training venv not found at ${VENV_TRAIN}. Run bash slurm/setup_pelle.sh first." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${VENV_TRAIN}/bin/activate"
}

print_job_header() {
    echo "============================================================"
    echo "Job ID       : ${SLURM_JOB_ID:-local} (Array: ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none})"
    echo "Job Name     : ${SLURM_JOB_NAME:-unknown}"
    echo "Host         : $(hostname)"
    echo "Start Time   : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Project Root : ${PROJECT_ROOT}"
    echo "HF_HOME      : ${HF_HOME}"
    echo "CUDA Devices : ${CUDA_VISIBLE_DEVICES:-unset}"
    echo "============================================================"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    fi
}
