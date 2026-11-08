#!/usr/bin/env bash
# Push EchoHate code + an entire data seed bucket (default seed_0_model) to hpclogin.
# Runs in a subshell — safe for Cursor/IDE terminals (do not `source hpc_defaults` by hand).
#
# Usage (from experiment/echohate):
#   bash job_submission/sync_seed_bucket_to_hpc_from_local.sh
#   bash job_submission/sync_seed_bucket_to_hpc_from_local.sh seed_0_model
#   DRY_RUN=1 bash job_submission/sync_seed_bucket_to_hpc_from_local.sh seed_0_model
#   HPC_RSYNC_USE_PLAIN_SSH=1 bash job_submission/sync_seed_bucket_to_hpc_from_local.sh
#
# Env: ECHOHATE_HPC_PROFILE=hpclogin_dgx  HPC_RA_RESOLVE=1  HPC_PASSWORD / SSHPASS (optional)
#      HPC_RSYNC_USE_PLAIN_SSH=1  — use SSH keys only (avoids sshpass+passphrase deadlock)
#      SYNC_CODE_ONLY=1           — skip data/ (code + job_submission only)
#
set -euo pipefail

unset HPC_HOST HPC_RA HPC_DIR 2>/dev/null || true
export ECHOHATE_HPC_PROFILE="${ECHOHATE_HPC_PROFILE:-hpclogin_dgx}"
export USE_LEGACY_HPC_LOGIN=1
export HPC_RA_RESOLVE="${HPC_RA_RESOLVE:-1}"

_BOOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_BOOT_DIR}/hpc_defaults_from_local.sh"
hpc_apply_ra_default_or_resolve

SEED_BUCKET="${1:-${ECHOHATE_DATA_SEED_DIR:-seed_0_model}}"
HPC_DIR="${HPC_RA}/experiment/echohate"
LOCAL_ECHOHATE="$(cd "${_BOOT_DIR}/.." && pwd)"
LOCAL_SEED="${LOCAL_ECHOHATE}/data/${SEED_BUCKET}"

if [[ ! -d "${LOCAL_SEED}" ]]; then
  echo "ERROR: missing ${LOCAL_SEED}" >&2
  exit 1
fi

RSYNC_EXTRA=()
if [[ "${DRY_RUN:-0}" = "1" ]]; then
  RSYNC_EXTRA+=(--dry-run)
  echo "[sync_seed_bucket] DRY_RUN=1"
fi

_RSYNC_FLAGS="${HPC_RSYNC_CODE_FLAGS:--avz}"
# --progress floods IDE terminals; use --info=progress2 in a real terminal if you want a bar.
if [[ "${HPC_RSYNC_SHOW_PROGRESS:-0}" = "1" ]]; then
  _RSYNC_FLAGS="${_RSYNC_FLAGS} --info=progress2"
fi

echo "=========================================="
echo "sync seed bucket → HPC"
echo "  Local:  ${LOCAL_SEED}"
echo "  Remote: ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/${SEED_BUCKET}/"
echo "  HPC_RA: ${HPC_RA}"
echo "=========================================="

hpc_mkdir_remote_from_local "${HPC_DIR}" "${HPC_DIR}/data" "${HPC_DIR}/data/${SEED_BUCKET}"

echo "[1/2] rsync code (exclude data/)..."
# shellcheck disable=SC2086
hpc_rsync_with_retry ${_RSYNC_FLAGS} ${RSYNC_EXTRA[@]+"${RSYNC_EXTRA[@]}"} \
  --exclude="data/" --exclude=".git/" --exclude="__pycache__/" --exclude=".DS_Store" \
  "${LOCAL_ECHOHATE}/" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/"

if [[ "${SYNC_CODE_ONLY:-0}" != "1" ]]; then
  echo "[2/2] rsync data/${SEED_BUCKET}/ ..."
  # shellcheck disable=SC2086
  hpc_rsync_with_retry ${_RSYNC_FLAGS} ${RSYNC_EXTRA[@]+"${RSYNC_EXTRA[@]}"} \
    --exclude='.DS_Store' \
    "${LOCAL_SEED}/" \
    "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/${SEED_BUCKET}/"
else
  echo "[2/2] SYNC_CODE_ONLY=1 — skip data/"
fi

echo "Done."
