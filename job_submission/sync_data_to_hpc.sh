#!/usr/bin/env bash
# Push echohate/data/ (or one dataset) to the HPC under ${HPC_RA}/experiment/echohate/data/.
# Uses rsync **without --delete**: files that exist only on the HPC are not removed.
#
# Usage (from RA repo root or anywhere with experiment/echohate or echohate):
#   bash job_submission/sync_data_to_hpc.sh
#   DATASET=HateXplain bash job_submission/sync_data_to_hpc.sh
#   DRY_RUN=1 bash job_submission/sync_data_to_hpc.sh
#
# Env (see job_submission/hpc_defaults_from_local.sh):
#   HPC_USER  HPC_HOST  HPC_RA  (optional: HPC_RA_RESOLVE=1 to set HPC_RA from remote $HOME/RA)
#
# Optional:
#   DATASET — if set, only syncs data/<DATASET>/ (not the whole data tree).
#   DRY_RUN=1 — rsync --dry-run
#   SYNC_EXCLUDE_HEAVY=1 — add excludes for large run dirs (same idea as some submit scripts).
#
set -euo pipefail

_BOOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_BOOT_DIR}/hpc_defaults_from_local.sh"
hpc_apply_ra_default_or_resolve
HPC_DIR="${HPC_RA}/experiment/echohate"

THIS_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This file lives in job_submission/; repo root scripts are one level up.
if [ -f "${THIS_SCRIPT}/../roberta_baseline.py" ]; then
  LOCAL_ECHOHATE="$(cd "${THIS_SCRIPT}/.." && pwd)"
elif [ -d "experiment/echohate" ]; then
  LOCAL_ECHOHATE="$(pwd)/experiment/echohate"
elif [ -d "echohate" ]; then
  LOCAL_ECHOHATE="$(pwd)/echohate"
else
  echo "ERROR: Could not find echohate (expected experiment/echohate or echohate under cwd)."
  exit 1
fi

LOCAL_DATA="${LOCAL_ECHOHATE}/data"
if [ ! -d "${LOCAL_DATA}" ]; then
  echo "ERROR: Missing local data dir: ${LOCAL_DATA}"
  exit 1
fi

# Note: with `set -u`, `"${empty_array[@]}"` can error on older bash (e.g. macOS 3.2) when empty.
RSYNC_EXTRA=()
if [ "${DRY_RUN:-0}" = "1" ]; then
  RSYNC_EXTRA+=(--dry-run)
  echo "[sync_data_to_hpc] DRY_RUN=1 (no files transferred)"
fi

EXCLUDES=(--exclude='.DS_Store')
if [ "${SYNC_EXCLUDE_HEAVY:-0}" = "1" ]; then
  EXCLUDES+=(
    --exclude='cue_distill_gridsearch*/'
    --exclude='roberta_baseline*/'
    --exclude='roberta_large_full_finetuned*/'
    --exclude='hatebert_baseline*/'
    --exclude='RESULTS_TABLE*'
    --exclude='*.html'
  )
  echo "[sync_data_to_hpc] SYNC_EXCLUDE_HEAVY=1: skipping some large run output patterns."
fi

echo "=========================================="
echo "echohate: sync data → HPC"
echo "  Local data:  ${LOCAL_DATA}"
echo "  Remote:      ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/"
echo "  HPC_RA:      ${HPC_RA}"
if [ -n "${DATASET:-}" ]; then
  echo "  Dataset:     ${DATASET} only"
else
  echo "  Dataset:     (all under data/)"
fi
echo "=========================================="

if [ -n "${DATASET:-}" ]; then
  if [ ! -d "${LOCAL_DATA}/${DATASET}" ]; then
    echo "ERROR: Missing ${LOCAL_DATA}/${DATASET}"
    exit 1
  fi
  hpc_mkdir_remote_from_local "${HPC_DIR}" "${HPC_DIR}/data" "${HPC_DIR}/data/${DATASET}"
  echo "Running rsync..."
  # shellcheck disable=SC2295
  hpc_rsync_with_retry -avz --progress ${RSYNC_EXTRA[@]+"${RSYNC_EXTRA[@]}"} "${EXCLUDES[@]}" \
    "${LOCAL_DATA}/${DATASET}/" \
    "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/${DATASET}/"
else
  hpc_mkdir_remote_from_local "${HPC_DIR}" "${HPC_DIR}/data"
  echo "Running rsync (full data/)..."
  # shellcheck disable=SC2295
  hpc_rsync_with_retry -avz --progress ${RSYNC_EXTRA[@]+"${RSYNC_EXTRA[@]}"} "${EXCLUDES[@]}" \
    "${LOCAL_DATA}/" \
    "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/"
fi

echo "Done."
