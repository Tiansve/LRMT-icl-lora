#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -c 3
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J lrmt_icl
#SBATCH -o slurm/logs/%x_%A_%a.out
#SBATCH -e slurm/logs/%x_%A_%a.err
#
# Stage 4: In-Context Learning (ICL) job array.
# Each array task processes ONE source language from SOURCE_LANGS[SLURM_ARRAY_TASK_ID]
# across all requested pools (default: 7 pools) and k in {5, 20, 50} (= 21 runs),
# loading the 8B model into vLLM only ONCE per language.
#
# Recommended staged submission on Pelle:
#   1. Pilot batch (first 4 languages: swh_Latn, kaz_Cyrl, amh_Ethi, sat_Olck):
#      sbatch -A <proj> --array=0-3 slurm/02_icl_array.sh
#
#   2. Remaining 8 languages (after verifying pilot outputs & scores):
#      sbatch -A <proj> --array=4-11 slurm/02_icl_array.sh
#
#   3. English control (Qwen3-8B -> eng_Latn):
#      sbatch -A <proj> --export=ALL,TGT=eng_Latn --array=0-3 slurm/02_icl_array.sh
#      sbatch -A <proj> --export=ALL,TGT=eng_Latn --array=4-11 slurm/02_icl_array.sh
#      (If time is tight per reduction plan #2, run seed 1 only:
#       sbatch -A <proj> --export=ALL,TGT=eng_Latn,POOLS="1:50 1:200 1:997" --array=0-11 slurm/02_icl_array.sh)
#
#   4. Extension (Llama-3.1-8B -> cmn_Hans):
#      sbatch -A <proj> --export=ALL,MODEL=llama3.1-8b --array=0-11 slurm/02_icl_array.sh

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
print_job_header
activate_infer

MODEL="${MODEL:-qwen3-8b}"
TGT="${TGT:-cmn_Hans}"
K_LIST="${K_LIST:-5 20 50}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID < 0 || TASK_ID >= ${#SOURCE_LANGS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range (0..$(( ${#SOURCE_LANGS[@]} - 1 )))" >&2
    exit 1
fi
SRC="${SOURCE_LANGS[$TASK_ID]}"

echo "Running ICL: model=${MODEL} src=${SRC} tgt=${TGT} k=[${K_LIST}] pools=[${POOLS:-all}]"

# shellcheck disable=SC2086
if [[ -n "${POOLS:-}" ]]; then
    python scripts/run_icl.py --model "${MODEL}" --langs "${SRC}" --tgt "${TGT}" --k ${K_LIST} --pools ${POOLS}
else
    python scripts/run_icl.py --model "${MODEL}" --langs "${SRC}" --tgt "${TGT}" --k ${K_LIST}
fi

echo "Finished ICL for ${SRC} at $(date '+%Y-%m-%d %H:%M:%S')"
