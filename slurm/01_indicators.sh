#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -c 3
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J lrmt_indicators
#SBATCH -o slurm/logs/%x_%j.out
#SBATCH -e slurm/logs/%x_%j.err
#
# Stage 2: Compute training-free indicators (Information Parity + Zero-Shot chrF/BLEU)
# for all 13 languages (12 sources + held-out Zulu) across both models (Qwen3-8B & Llama-3.1-8B),
# plus the English control zero-shot baseline for Qwen3-8B.
#
# Submit from repository root:
#   sbatch -A <your_uppmax_project_id> slurm/01_indicators.sh
# (To use H100 if the L40S queue is busy: sbatch -A <proj> --gpus=h100:1 slurm/01_indicators.sh)

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
print_job_header
activate_infer

echo "[1/4] Computing Information Parity (dev, 997 sentences, English & Chinese pivots)..."
python scripts/info_parity.py --models qwen3-8b llama3.1-8b --langs all

echo "[2/4] Running Zero-Shot (k=0) into Simplified Chinese (devtest, 1012 sentences)..."
python scripts/run_icl.py --model qwen3-8b --langs all-with-heldout --tgt cmn_Hans --k 0
python scripts/run_icl.py --model llama3.1-8b --langs all-with-heldout --tgt cmn_Hans --k 0

echo "[3/4] Running Zero-Shot (k=0) into English control for Qwen3-8B..."
python scripts/run_icl.py --model qwen3-8b --langs all-with-heldout --tgt eng_Latn --k 0

echo "[4/4] Scoring all outputs and updating indicators/zeroshot.csv..."
python scripts/score.py

echo "Finished at $(date '+%Y-%m-%d %H:%M:%S')"
