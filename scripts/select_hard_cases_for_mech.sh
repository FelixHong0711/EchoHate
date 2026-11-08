#!/usr/bin/env bash
# Select hard training cases for mech-only encoder workflows.
#
# Wrapper around: python -m pipeline.stage1.select_hard_cases_for_mech
#
# Examples (from experiment/echohate):
#
#   # 50% stratified random (rand50) — all 7 DS, wCL+woCL, mirror for toy_prompt
#   SEED_DIR=eacl_seed0_seq METHOD=rand50 MIRROR_GMM=1 UPDATE_MANIFEST=1 \
#     bash scripts/select_hard_cases_for_mech.sh
#
#   # Top-25% variability — wCL only, do NOT mirror (keep var25 side by side)
#   SEED_DIR=eacl_seed0_seq METHOD=var25 BASELINES=roberta_large_baseline_wCL \
#     bash scripts/select_hard_cases_for_mech.sh
#
#   # Single dataset
#   SEED_DIR=eacl_seed0_seq METHOD=rand50 DATASETS=IHC \
#     MIRROR_GMM=1 bash scripts/select_hard_cases_for_mech.sh
#
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ECHOHATE_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${ECHOHATE_ROOT}"

METHOD="${METHOD:-rand50}"
SEED_DIR="${SEED_DIR:-eacl_seed0_seq}"
DATA_ROOT="${DATA_ROOT:-data}"
DATASETS="${DATASETS:-IHC,DynaHate,HatEval,Ethos,drop_offensive,toxigen_new,SBIC}"
BASELINES="${BASELINES:-roberta_large_baseline_wCL,roberta_large_baseline_woCL}"
SPLIT_SUBDIR="${SPLIT_SUBDIR:-splitted_data_80_10_10}"
SAMPLE_FRAC="${SAMPLE_FRAC:-0.5}"
TOP_FRAC="${TOP_FRAC:-0.25}"
RANDOM_STATE="${RANDOM_STATE:-}"

_extra=()
if [[ "${MIRROR_GMM:-0}" == "1" ]]; then
  _extra+=(--mirror-selection-gmm)
fi
if [[ "${UPDATE_MANIFEST:-0}" == "1" ]]; then
  _extra+=(--update-seed-manifest)
fi
if [[ -n "${RUN_DIR:-}" ]]; then
  _extra+=(--run-dir "${RUN_DIR}")
fi
if [[ -n "${TRAIN_CSV:-}" ]]; then
  _extra+=(--train-csv "${TRAIN_CSV}")
fi
if [[ -n "${RANDOM_STATE}" ]]; then
  _extra+=(--random-state "${RANDOM_STATE}")
fi

PYTHONPATH="${ECHOHATE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 -m pipeline.stage1.select_hard_cases_for_mech \
  --method "${METHOD}" \
  --data-root "${DATA_ROOT}" \
  --seed-dir "${SEED_DIR}" \
  --datasets "${DATASETS}" \
  --baselines "${BASELINES}" \
  --split-subdir "${SPLIT_SUBDIR}" \
  --sample-frac "${SAMPLE_FRAC}" \
  --top-frac "${TOP_FRAC}" \
  ${_extra[@]+"${_extra[@]}"}
