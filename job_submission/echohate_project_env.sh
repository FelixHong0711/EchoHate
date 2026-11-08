#!/usr/bin/env bash
# shellcheck shell=bash
# Load all defaults from config/echohate_project_config.json into the shell.
#
# Sets (when unset): RANDOM_STATE, SEED, stage seeds, split fractions, ECHOHATE_DATA_SEED_DIR,
# DATASET, AMPLEHATE_SEED_LIST, training hyperparameters (unless ECHOHATE_SKIP_PROJECT_TRAINING=1).
#
# Usage:
#   export ECHOHATE_ROOT=/path/to/experiment/echohate   # optional
#   source "${ECHOHATE_ROOT}/job_submission/echohate_project_env.sh"
#
# Skip entirely: ECHOHATE_SKIP_PROJECT_ENV=1
# Skip training keys only: ECHOHATE_SKIP_PROJECT_TRAINING=1

if [ "${ECHOHATE_SKIP_PROJECT_ENV:-0}" = "1" ]; then
  return 0 2>/dev/null || exit 0
fi

_echohate_pe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ECHOHATE_ROOT="${ECHOHATE_ROOT:-$(cd "${_echohate_pe_dir}/.." && pwd)}"
export ECHOHATE_ROOT

# shellcheck source=/dev/null
source "${_echohate_pe_dir}/echohate_seed_env.sh"

_cfg="${ECHOHATE_ROOT}/config/echohate_project_config.json"
if [ ! -f "${_cfg}" ]; then
  echo "[echohate_project_env] WARN: missing ${_cfg}; seed env only" >&2
  return 0 2>/dev/null || exit 0
fi

if ! eval "$(cd "${ECHOHATE_ROOT}" && python3 -m pipeline.common.project_config --export-shell 2>/dev/null)"; then
  echo "[echohate_project_env] WARN: --export-shell failed; seed env only" >&2
elif [ "${ECHOHATE_SKIP_PROJECT_TRAINING:-0}" != "1" ]; then
  eval "$(cd "${ECHOHATE_ROOT}" && python3 -m pipeline.common.project_config --export-training-shell 2>/dev/null)" || true
fi

# shellcheck source=/dev/null
[ -f "${_echohate_pe_dir}/echohate_hf_token_env.sh" ] && source "${_echohate_pe_dir}/echohate_hf_token_env.sh"
if command -v echohate_export_hf_token_from_secrets >/dev/null 2>&1; then
  echohate_export_hf_token_from_secrets "${ECHOHATE_ROOT}"
fi
