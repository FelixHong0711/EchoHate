#!/usr/bin/env bash
# hpclogin DGX wrapper → submit_roberta_fft_mech_cue_post_llm_from_local.sh (mech-only).
#
#   Mech alone:  APPROACH=mech  ENCODER_RUN_TAG=_mech_only
#   APPROACH=mech only
#
# Same underlying script as `submit_roberta_fft_mech_cue_post_llm_from_local.sh` on
# **legacy** Deakin login **hpclogin…** + Slurm **dgx**.
#
# DGX pool on hpclogin: **only hpc-dgx-b-2,hpc-dgx-b-3** (default here and in hpc_defaults). **hpc-dgx-b-3** is
# stripped in post_llm unless SBATCH_ALLOW_DGX_B3=1. Each sbatch uses `--nodes=1 --ntasks=1 --nodelist=<comma pool>`
# so Slurm allocates **one** host from {b-2,b-3} (a bare comma nodelist without --nodes=1 can look like **2 nodes**).
#
# Usage (from echohate root or RA root that contains experiment/echohate):
#   cd .../experiment/echohate
#   export HPC_RA=/home/hohuu/RA          # or: HPC_RA_RESOLVE=1
#   ONLY_DATASETS=IHC APPROACH=mech ... \
#   bash job_submission/stage2/cue_mech_attnmask/submit_roberta_fft_mech_cue_hpclogin_dgx_from_local.sh
#
# Override defaults:
#   APPROACH=mech
#   SBATCH_NODES='hpc-dgx-b-2' HPC_ALLOW_DGX_B1=1 SBATCH_PARTITION=dgx SBATCH_MEM=128G ...
#   CONDA_ENV=rag_v8_gpu  (must exist on hpclogin)
#
set -euo pipefail

# Preserve caller HPC_RA / HPC_RA_RESOLVE (mech-only wrapper sets these).
# Only unset HPC_HOST/HPC_DIR so hpc_defaults can rebuild them for hpclogin_dgx.
unset HPC_HOST HPC_DIR 2>/dev/null || true
export ECHOHATE_HPC_PROFILE=hpclogin_dgx
export USE_LEGACY_HPC_LOGIN=1
export HPC_RA_RESOLVE="${HPC_RA_RESOLVE:-0}"
export HPC_RA="${HPC_RA:-/scratch/hohuu/RA}"
export HPLOGIN_DGX_NODE_CSV="${HPLOGIN_DGX_NODE_CSV:-hpc-dgx-b-2,hpc-dgx-b-3}"
export SBATCH_NODES="${SBATCH_NODES:-${HPLOGIN_DGX_NODE_CSV}}"
export HPLOGIN_DGX_ROUND_ROBIN="${HPLOGIN_DGX_ROUND_ROBIN:-0}"
export HPLOGIN_DGX_PROBE_CUDA_NODES="${HPLOGIN_DGX_PROBE_CUDA_NODES:-1}"
export APPROACH="${APPROACH:-mech}"

_BOOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${_BOOT_DIR}/submit_roberta_fft_mech_cue_post_llm_from_local.sh" "$@"
