#!/usr/bin/env bash
# seed_0_mech: RoBERTa-large FFT baselines (sibling wCL / woCL folders).
#
# Same splits as seed_0_comparator, RANDOM_STATE=0. Train both CL modes fresh
# (copied comparator CSVs have no checkpoints).
#
# Usage (from experiment/echohate):
#   unset HPC_SSH_CONN_OPTS
#   export HPC_RA=/home/remote/hohuu/RA HPC_RA_RESOLVE=0
#   bash job_submission/stage2/roberta_baseline/submit_roberta_baseline_seed0_mech_hpclogin_dgx_from_local.sh
#
# One mode:
#   CL_MODES=wCL bash .../submit_roberta_baseline_seed0_mech_hpclogin_dgx_from_local.sh
# SBIC (5-way offensive) is off by default:
#   DATASETS=IHC,DynaHate,HatEval,drop_offensive,toxigen_new,Ethos,SBIC bash ...
#
set -euo pipefail

unset HPC_HOST HPC_DIR 2>/dev/null || true
# Ambient leaks: seed_0_comparator / seed_42_model / RANDOM_STATE=42 must not win.
unset ECHOHATE_DATA_SEED_DIR ECHOHATE_ROBERTA_BASELINE_DIR OUTPUT_DIR OUTPUT_TAG 2>/dev/null || true

export ECHOHATE_HPC_PROFILE=hpclogin_dgx
export USE_LEGACY_HPC_LOGIN=1
export HPC_RA_RESOLVE="${HPC_RA_RESOLVE:-0}"
export HPC_RA="${HPC_RA:-/scratch/hohuu/RA}"
export HPC_SSHPASS_FOR_MKDIR="${HPC_SSHPASS_FOR_MKDIR:-1}"

_BOOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ECHOHATE_DATA_SEED_DIR=seed_0_mech
export ECHOHATE_SPLITTED_DATA_SUBDIR="${ECHOHATE_SPLITTED_DATA_SUBDIR:-splitted_data_80_10_10}"
export DATASETS="${DATASETS:-IHC,DynaHate,HatEval,drop_offensive,toxigen_new,Ethos}"
export CONDA_ENV="${CONDA_ENV:-rag_v8_gpu}"
export RANDOM_STATE=0
export TRACK_DYNAMICS=1

unset HPLOGIN_DGX_NODE_CSV SBATCH_NODES SBATCH_EXTRA 2>/dev/null || true
export SBATCH_ALLOW_DGX_B3="${SBATCH_ALLOW_DGX_B3:-1}"
export HPLOGIN_DGX_NODE_CSV="${HPLOGIN_DGX_NODE_CSV:-hpc-dgx-b-2,hpc-dgx-b-3}"
export SBATCH_NODES="${SBATCH_NODES:-${HPLOGIN_DGX_NODE_CSV}}"

CL_MODES="${CL_MODES:-woCL,wCL}"

echo "Submitting RoBERTa FFT baselines for ${ECHOHATE_DATA_SEED_DIR} (RANDOM_STATE=${RANDOM_STATE}):"
echo "  DATASETS=${DATASETS}"
echo "  CL_MODES=${CL_MODES}"
echo "  SBATCH_NODES=${SBATCH_NODES}  SBATCH_ALLOW_DGX_B3=${SBATCH_ALLOW_DGX_B3}"

IFS=',' read -r -a _modes <<< "${CL_MODES}"
for _mode in "${_modes[@]}"; do
  _mode="${_mode// /}"
  [[ -n "${_mode}" ]] || continue
  unset DISABLE_CE_CONTRASTIVE CE_CONTRASTIVE_LAMBDA CE_CONTRASTIVE_LAMBDA_EXPLICIT
  unset ECHOHATE_ROBERTA_BASELINE_DIR
  case "${_mode}" in
    woCL)
      export DISABLE_CE_CONTRASTIVE=1
      export ECHOHATE_ROBERTA_BASELINE_DIR=roberta_large_baseline_woCL
      ;;
    wCL)
      export DISABLE_CE_CONTRASTIVE=0
      export CE_CONTRASTIVE_LAMBDA_EXPLICIT=1
      export CE_CONTRASTIVE_LAMBDA=0.5
      export ECHOHATE_ROBERTA_BASELINE_DIR=roberta_large_baseline_wCL
      ;;
    *)
      echo "ERROR: CL_MODES entry must be woCL or wCL, got: ${_mode}" >&2
      exit 1
      ;;
  esac
  echo ""
  echo "===== ${_mode} → data/${ECHOHATE_DATA_SEED_DIR}/<DS>/${ECHOHATE_ROBERTA_BASELINE_DIR} ====="
  bash "${_BOOT_DIR}/submit_roberta_baseline_from_local.sh"
done
