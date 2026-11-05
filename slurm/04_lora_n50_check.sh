#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -c 3
#SBATCH --mem=32G
#SBATCH -t 03:30:00
#SBATCH -J lrmt_lora_n50
#SBATCH -o slurm/logs/%x_%j.out
#SBATCH -e slurm/logs/%x_%j.err
#
# Stage 5 (step 2): First-batch QLoRA sanity check across all 12 source languages
# at N=50, seed=1 ("先全部语言 N = 50 种子 1 一批，检查后再提交其余").
# Trains all 12 adapters in one 4-bit model load, runs vLLM inference on all 12,
# and updates scores/scores.csv.
#
# Submit from repository root:
#   sbatch -A <proj> slurm/04_lora_n50_check.sh

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
print_job_header

MODEL="${MODEL:-qwen3-8b}"
TGT="${TGT:-cmn_Hans}"

echo "[1/3] Training N=50 seed=1 QLoRA adapters for all 12 source languages (${MODEL} -> ${TGT})..."
activate_train
python scripts/train_lora.py --model "${MODEL}" --langs sources --tgt "${TGT}" --pools 1:50
deactivate

echo "[2/3] Running vLLM inference for N=50 seed=1 LoRA adapters..."
activate_infer
python scripts/run_lora.py --model "${MODEL}" --langs sources --tgt "${TGT}" --pools 1:50

echo "[3/3] Updating scores/scores.csv..."
python scripts/score.py
deactivate

echo "Finished N=50 seed=1 LoRA check at $(date '+%Y-%m-%d %H:%M:%S')"
