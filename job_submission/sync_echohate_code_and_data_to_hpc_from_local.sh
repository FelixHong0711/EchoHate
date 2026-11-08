#!/usr/bin/env bash
# Sync EchoHate **code** (excluding top-level data/) plus **data/** (subtree or entire tree) to the cluster.
# Same rsync layout as submit_roberta_fullft_ihc_supcon_lambda_grid_from_local.sh (no Slurm).
#
# Safety: rsync does **not** use --delete. Extra files on the HPC that are absent locally are left
# untouched (only adds/updates from your machine).
#
# Usage (from RA repo root or from experiment/echohate):
#   bash job_submission/sync_echohate_code_and_data_to_hpc_from_local.sh
#   SYNC_ALL_DATA=1 bash job_submission/sync_echohate_code_and_data_to_hpc_from_local.sh
#   DATASET=IHC ECHOHATE_DATA_SEED_DIR=seed_0 bash job_submission/sync_echohate_code_and_data_to_hpc_from_local.sh
#   bash job_submission/sync_echohate_code_and_data_to_hpc_from_local.sh IHC seed_0
#
# Env (see job_submission/hpc_defaults_from_local.sh):
#   HPC_USER  HPC_HOST  HPC_RA  (optional: HPC_RA_RESOLVE=1)
#   hpc2: export ECHOHATE_HPC_PROFILE=hpc2  (or export HPC_HOST=hpc2.it.deakin.edu.au before running)
#   unset ECHOHATE_HPC_PROFILE alone still defaults to hpclogin_dgx — not hpc2.
#   SYNC_FULL=1         one rsync of entire echohate/ (code + data/ + job_submission/; excludes .git, __pycache__)
#   SYNC_ALL_DATA=1     two-step: code (no data/) then all of data/
#   SYNC_AMPLEHATE=1    also rsync experiment/AmpleHate → ${HPC_DIR}/AmpleHate (sibling repo layout)
#   DATASET              default IHC (ignored when SYNC_ALL_DATA=1)
#   ECHOHATE_DATA_SEED_DIR optional, e.g. seed_0 → sync data/seed_0/<DATASET>/
#   DRY_RUN=1            rsync --dry-run
#
set -euo pipefail

_BOOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_BOOT_DIR}/hpc_defaults_from_local.sh"
hpc_apply_ra_default_or_resolve
HPC_DIR="${HPC_RA}/experiment/echohate"

if [[ "${SYNC_ALL_DATA:-0}" != "1" ]]; then
  if [[ -n "${1:-}" && "${1}" != *=* ]]; then DATASET="$1"; shift; fi
  if [[ -n "${1:-}" && "${1}" != *=* ]]; then ECHOHATE_DATA_SEED_DIR="$1"; shift; fi
fi

DATASET="${DATASET:-IHC}"
ECHOHATE_DATA_SEED_DIR="${ECHOHATE_DATA_SEED_DIR:-}"

THIS_SCRIPT="${_BOOT_DIR}"
_co2_root() {
  local root="${THIS_SCRIPT}/.."
  if [ -f "${root}/pipeline/stage2/roberta_baseline.py" ] \
    || [ -f "${root}/README.md" ] \
    || [ -f "${root}/config/echohate_project_config.json" ]; then
    cd "${root}" && pwd
    return 0
  fi
  return 1
}
if _r="$(_co2_root 2>/dev/null)"; then
  LOCAL_ECHOHATE="${_r}"
elif [ -d "experiment/echohate" ]; then
  LOCAL_ECHOHATE="$(cd "experiment/echohate" && pwd)"
elif [ -d "echohate" ]; then
  LOCAL_ECHOHATE="$(cd "echohate" && pwd)"
else
  echo "ERROR: Run from RA repo root or experiment/echohate."
  exit 1
fi

if [ "${SYNC_ALL_DATA:-0}" = "1" ]; then
  if [ -n "${ECHOHATE_DATA_SEED_DIR}" ]; then
    # Prefer seed subtree (avoids syncing entire local data volume / symlink target).
    LOCAL_DATA_REL="data/${ECHOHATE_DATA_SEED_DIR}"
  else
    LOCAL_DATA_REL="data"
  fi
else
  LOCAL_DATA_REL="data/${DATASET}"
  if [ -n "${ECHOHATE_DATA_SEED_DIR}" ]; then
    LOCAL_DATA_REL="data/${ECHOHATE_DATA_SEED_DIR}/${DATASET}"
  fi
fi

RSYNC_EXTRA=()
if [ "${DRY_RUN:-0}" = "1" ]; then
  RSYNC_EXTRA+=(--dry-run)
  echo "[sync_echohate_code_and_data] DRY_RUN=1"
fi

_RSYNC_EXCLUDES=(
  --exclude=".git/"
  --exclude="__pycache__/"
  --exclude="*.pyc"
  --exclude=".DS_Store"
)

if [ "${SYNC_FULL:-0}" = "1" ]; then
  echo "=========================================="
  echo "EchoHate: SYNC_FULL=1 (entire experiment/echohate → HPC)"
  echo "  Local:  ${LOCAL_ECHOHATE}"
  echo "  Remote: ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/"
  echo "=========================================="
  hpc_mkdir_remote_from_local "${HPC_DIR}"
  hpc_rsync_with_retry -avz --progress ${RSYNC_EXTRA[@]+"${RSYNC_EXTRA[@]}"} \
    "${_RSYNC_EXCLUDES[@]}" \
    "${LOCAL_ECHOHATE}/" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/"
  if [ "${SYNC_AMPLEHATE:-1}" = "1" ]; then
    _ah="${LOCAL_ECHOHATE}/../AmpleHate"
    if [ -f "${_ah}/train.py" ]; then
      echo "[SYNC_AMPLEHATE] rsync ${_ah} → AmpleHate/"
      hpc_mkdir_remote_from_local "${HPC_DIR}/AmpleHate"
      hpc_rsync_with_retry -avz --progress ${RSYNC_EXTRA[@]+"${RSYNC_EXTRA[@]}"} \
        "${_RSYNC_EXCLUDES[@]}" --exclude="save/" \
        "${_ah}/" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/AmpleHate/"
    else
      echo "[SYNC_AMPLEHATE] skip (no ${_ah}/train.py)"
    fi
  fi
  echo "Done (SYNC_FULL)."
  exit 0
fi

if [ ! -d "${LOCAL_ECHOHATE}/${LOCAL_DATA_REL}" ]; then
  echo "ERROR: Missing local dir: ${LOCAL_ECHOHATE}/${LOCAL_DATA_REL}"
  exit 1
fi

echo "=========================================="
echo "EchoHate: sync code + data → HPC"
echo "  Local EchoHate root: ${LOCAL_ECHOHATE}"
echo "  Data rel:     ${LOCAL_DATA_REL}"
if [ "${SYNC_ALL_DATA:-0}" = "1" ]; then
  echo "  Mode:         SYNC_ALL_DATA=1 → ${LOCAL_DATA_REL}/"
fi
echo "  Remote:       ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/"
echo "  HPC_RA:       ${HPC_RA}"
echo "  Note:         no --delete (HPC-only files are kept)"
echo "=========================================="

hpc_mkdir_remote_from_local "${HPC_DIR}" "${HPC_DIR}/${LOCAL_DATA_REL}"

echo "[1/2] rsync code (exclude top-level data/ — also exclude symlink named data)..."
# shellcheck disable=SC2295
hpc_rsync_with_retry -avz --progress ${RSYNC_EXTRA[@]+"${RSYNC_EXTRA[@]}"} \
  --exclude="data" --exclude="data/" --exclude=".git/" --exclude="__pycache__/" --exclude=".DS_Store" \
  "${LOCAL_ECHOHATE}/" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/"

echo "[2/2] rsync ${LOCAL_DATA_REL}/ ..."
# shellcheck disable=SC2295
hpc_rsync_with_retry -avz --progress ${RSYNC_EXTRA[@]+"${RSYNC_EXTRA[@]}"} \
  --exclude='.DS_Store' \
  "${LOCAL_ECHOHATE}/${LOCAL_DATA_REL}/" \
  "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/${LOCAL_DATA_REL}/"

echo "Done."
