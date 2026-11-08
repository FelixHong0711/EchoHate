#!/usr/bin/env bash
# Canonical shared training hyperparameters for EchoHate **fine-tuning** jobs (RoBERTa-family:
# baseline, warmstart, encoder distill, HateBERT baseline, roberta-large full fine-tune).
#
# Source from Slurm/submit scripts after `cd` to echohate root (or from submit_* after resolving paths):
#   # shellcheck source=/dev/null
#   source "${ECHOHATE_ROOT}/job_submission/training_hyper_defaults.sh"
#
# RNG defaults come from config/echohate_project_config.json (via echohate_seed_env.sh → pipeline.common.project_config).
# Training numeric defaults can also be set from the same JSON (see --export-training-shell below).

_RTH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ECHOHATE_ROOT="${ECHOHATE_ROOT:-$(cd "${_RTH_SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=/dev/null
source "${_RTH_SCRIPT_DIR}/echohate_project_env.sh"
#
# Override any variable via `sbatch --export=...` or your shell before sbatch.
#
# Does **not** set: DATASET, MODEL_NAME, OUT_SUBDIR (job-specific — set after sourcing).
#
# RoBERTa-large **full** fine-tune (encoder + head): use a conservative LR + warmup **before** sourcing
# this file so the defaults below do not clobber them, e.g. in hybrid/submit_roberta_large_full_finetune_from_local.sh:
#   : "${LEARNING_RATE:=3e-5}"
#   : "${WARMUP_RATIO:=0.06}"
#   source .../training_hyper_defaults.sh
# (Typical BERT/RoBERTa fine-tuning range is ~1e-5–5e-5; 1e-4 full-FT can diverge / collapse logits.)

NUM_EPOCHS="${NUM_EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
MAX_LENGTH="${MAX_LENGTH:-256}"
POS_WEIGHT_MODE="${POS_WEIGHT_MODE:-none}"
CLASS_WEIGHT_TARGET="${CLASS_WEIGHT_TARGET:-pos}"
PAD_TO_MAX_LENGTH="${PAD_TO_MAX_LENGTH:-1}"
# Match metric_summary default: eval each epoch enabled unless explicitly overridden.
NO_EVAL_EACH_EPOCH="${NO_EVAL_EACH_EPOCH:-0}"

# Train/validation/test CSVs live under data/<dataset>/splitted_data_80_10_10/ (see pipeline/common/split_layout.py;
# stratified splits from python -m pipeline.preprocessing split).
#
# -----------------------------------------------------------------------------
# Sync results from HPC → local (after jobs finish). Submit scripts print a filled-in
# rsync when you run without --wait. Use your HPC_USER, HPC_HOST, HPC_DIR (see
# job_submission/hpc_defaults_from_local.sh) and paths below.
#
# RoBERTa / HateBERT baseline (metrics.json only):
#   rsync -avz --progress --include='metrics.json' --exclude='*' \
#     "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/DATASET/OUT_SUBDIR/" \
#     "LOCAL_ECHOHATE/data/DATASET/OUT_SUBDIR/"
#
# RoBERTa-large full fine-tune (metric_summary + metrics):
#   rsync -avz --progress --include='metric_summary.json' --include='metrics.json' --exclude='*' \
#     "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/DATASET/OUT_SUBDIR/" \
#     "LOCAL_ECHOHATE/data/DATASET/OUT_SUBDIR/"
#
# Warmstart (skip heavy checkpoints):
#   rsync -avz --progress --exclude='model/' --exclude='*.pt' \
#     "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/DATASET/warmstart_result/" \
#     "LOCAL_ECHOHATE/data/DATASET/warmstart_result/"
#
# encoder distill grid (nested lm_* runs):
#   rsync -avz --progress --include='*/' --include='metrics.json' --exclude='*' \
#     "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/DATASET/OUT_ROOT_SUBDIR/" \
#     "LOCAL_ECHOHATE/data/DATASET/OUT_ROOT_SUBDIR/"
