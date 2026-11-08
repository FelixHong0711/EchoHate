#!/usr/bin/env bash
# shellcheck shell=bash
# Load HF token via pipeline.common.hf_auth and optionally rsync secrets/hf_token to HPC.
#
#   echohate_export_hf_token_from_secrets [ECHOHATE_ROOT]
#   echohate_sync_hf_token_to_hpc [ECHOHATE_ROOT] [HPC_DIR]
#   echohate_hf_token_export_suffix   # prints ,HF_TOKEN=... for sbatch --export
#
# Env:
#   ECHOHATE_SKIP_HF_TOKEN_RSYNC=1  skip rsync (token still loaded locally for --export)
#   ECHOHATE_SYNC_HF_TOKEN=1        default: sync when echohate_sync_hf_token_to_hpc is called
#   ECHOHATE_PYTHON                 python for token load (default: python3)

echohate_export_hf_token_from_secrets() {
  local root="${1:-${ECHOHATE_ROOT:-${LOCAL_ECHOHATE:-}}}"
  if [ -z "${root}" ] || [ ! -d "${root}" ]; then
    return 0
  fi
  if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    return 0
  fi
  local py="${ECHOHATE_PYTHON:-python3}"
  if ! command -v "${py}" >/dev/null 2>&1; then
    return 0
  fi
  local out
  out="$(
    ECHOHATE_ROOT="${root}" PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}" "${py}" - <<'PY'
import os
import shlex
import sys

root = os.environ["ECHOHATE_ROOT"]
sys.path.insert(0, root)
from pipeline.common.hf_auth import ensure_hf_token_env

tok = ensure_hf_token_env()
if tok:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        print(f"export {key}={shlex.quote(tok)}")
PY
  )" 2>/dev/null || true
  if [ -n "${out}" ]; then
    # shellcheck disable=SC2086
    eval "${out}"
  fi
}

echohate_sync_hf_token_to_hpc() {
  local root="${1:-${ECHOHATE_ROOT:-${LOCAL_ECHOHATE:-}}}"
  local hpc_dir="${2:-${HPC_DIR:-${HPC_RA:+${HPC_RA}/experiment/echohate}}}"
  if [ "${ECHOHATE_SKIP_HF_TOKEN_RSYNC:-0}" = "1" ]; then
    return 0
  fi
  if [ "${ECHOHATE_SYNC_HF_TOKEN:-1}" != "1" ]; then
    return 0
  fi
  if [ -z "${root}" ] || [ ! -f "${root}/secrets/hf_token" ]; then
    return 0
  fi
  if [ -z "${HPC_USER:-}" ] || [ -z "${HPC_HOST:-}" ] || [ -z "${hpc_dir}" ]; then
    echo "[echohate_hf_token] skip rsync (HPC_USER/HPC_HOST/HPC_DIR unset)" >&2
    return 0
  fi
  if ! command -v hpc_mkdir_remote_from_local >/dev/null 2>&1; then
    return 0
  fi
  echohate_export_hf_token_from_secrets "${root}"
  hpc_mkdir_remote_from_local "${hpc_dir}" "${hpc_dir}/secrets"
  hpc_rsync_with_retry -avz "${root}/secrets/hf_token" \
    "${HPC_USER}@${HPC_HOST}:${hpc_dir}/secrets/hf_token"
}

# Append HF_TOKEN to Slurm --export=... CSV (empty when unset).
echohate_hf_token_export_suffix() {
  if [ -n "${HF_TOKEN:-}" ]; then
    printf '%s' ",HF_TOKEN=${HF_TOKEN}"
  fi
}
