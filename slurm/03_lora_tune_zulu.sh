#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -c 3
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J lrmt_lora_zulu
#SBATCH -o slurm/logs/%x_%j.out
#SBATCH -e slurm/logs/%x_%j.err
#
# Stage 5 (step 1): Held-out QLoRA hyperparameter search on Zulu (zul_Latn).
# Trains the 2x2 grid (learning_rate in {1e-4, 2e-4} x rank in {8, 16}) across
# pools 1:50, 1:200, 1:997 (12 adapters total) using .venv-train, then evaluates
# all 12 adapters on devtest using vLLM in .venv-infer and writes scores/lora_tuning.csv.
#
# Submit from repository root:
#   sbatch -A <proj> slurm/03_lora_tune_zulu.sh

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
print_job_header

MODEL="${MODEL:-qwen3-8b}"
TGT="${TGT:-cmn_Hans}"
POOLS="${POOLS:-1:50 1:200 1:997}"

echo "[1/2] Training Zulu QLoRA hyperparameter grid (${MODEL} -> ${TGT}, pools: ${POOLS})..."
activate_train
# shellcheck disable=SC2086
python scripts/train_lora.py --model "${MODEL}" --langs zul_Latn --tgt "${TGT}" --pools ${POOLS} --grid-search
deactivate

echo "[2/2] Evaluating Zulu QLoRA grid with vLLM and ranking hyperparameters..."
activate_infer
# shellcheck disable=SC2086
python scripts/run_lora.py --model "${MODEL}" --langs zul_Latn --tgt "${TGT}" --pools ${POOLS} --grid-search
deactivate

echo "Finished Zulu hyperparameter search at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Inspect scores/lora_tuning.csv and update rank / learning_rate in configs/lora.yaml if needed."
