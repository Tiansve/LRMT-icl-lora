#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -c 3
#SBATCH --mem=32G
#SBATCH -t 03:30:00
#SBATCH -J lrmt_lora
#SBATCH -o slurm/logs/%x_%A_%a.out
#SBATCH -e slurm/logs/%x_%A_%a.err
#
# Stage 5 (step 3): Full QLoRA training + vLLM inference job array.
# Each array task processes ONE source language from SOURCE_LANGS[SLURM_ARRAY_TASK_ID]
# across all requested pools (default: all 7 pools; 1:50 is skipped automatically if
# already completed by 04_lora_n50_check.sh), then immediately runs vLLM inference
# for all 7 adapters of that language.
#
# Usage on Pelle:
#   1. Qwen3-8B Chinese main experiment (all 12 languages):
#      sbatch -A <proj> --array=0-11 slurm/05_lora_array.sh
#
#   2. Qwen3-8B English control:
#      sbatch -A <proj> --export=ALL,TGT=eng_Latn --array=0-11 slurm/05_lora_array.sh
#      (Or seed-1-only per reduction plan #3:
#       sbatch -A <proj> --export=ALL,TGT=eng_Latn,POOLS="1:50 1:200 1:997" --array=0-11 slurm/05_lora_array.sh)
#
#   3. Extension (Llama-3.1-8B Chinese):
#      sbatch -A <proj> --export=ALL,MODEL=llama3.1-8b --array=0-11 slurm/05_lora_array.sh

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
print_job_header

MODEL="${MODEL:-qwen3-8b}"
TGT="${TGT:-cmn_Hans}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID < 0 || TASK_ID >= ${#SOURCE_LANGS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range (0..$(( ${#SOURCE_LANGS[@]} - 1 )))" >&2
    exit 1
fi
SRC="${SOURCE_LANGS[$TASK_ID]}"

echo "[1/2] Training QLoRA adapters: model=${MODEL} src=${SRC} tgt=${TGT} pools=[${POOLS:-all}]"
activate_train
# shellcheck disable=SC2086
if [[ -n "${POOLS:-}" ]]; then
    python scripts/train_lora.py --model "${MODEL}" --langs "${SRC}" --tgt "${TGT}" --pools ${POOLS}
else
    python scripts/train_lora.py --model "${MODEL}" --langs "${SRC}" --tgt "${TGT}"
fi
deactivate

echo "[2/2] Running vLLM inference for trained adapters: model=${MODEL} src=${SRC} tgt=${TGT}"
activate_infer
# shellcheck disable=SC2086
if [[ -n "${POOLS:-}" ]]; then
    python scripts/run_lora.py --model "${MODEL}" --langs "${SRC}" --tgt "${TGT}" --pools ${POOLS}
else
    python scripts/run_lora.py --model "${MODEL}" --langs "${SRC}" --tgt "${TGT}"
fi
deactivate

echo "Finished LoRA training + inference for ${SRC} at $(date '+%Y-%m-%d %H:%M:%S')"
