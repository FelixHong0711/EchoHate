#!/usr/bin/env bash
# shellcheck shell=bash
# Align DATA_ROOT with ECHOHATE_DATA_SEED_DIR and check OUTPUT_DIR uses the same seed bucket.
#
# Usage (after setting/exporting ECHOHATE_DATA_SEED_DIR):
#   # shellcheck source=/dev/null
#   source "${ECHOHATE_ROOT}/job_submission/echohate_data_paths.sh"
#   echohate_align_data_root
#   echohate_assert_seed_paths_consistent "${OUTPUT_DIR:-}"

echohate_align_data_root() {
  local seed="${ECHOHATE_DATA_SEED_DIR:-}"
  if [[ -z "${seed}" ]]; then
    echo "[echohate_data_paths] WARN: ECHOHATE_DATA_SEED_DIR unset; DATA_ROOT unchanged (${DATA_ROOT:-<unset>})" >&2
    return 0
  fi
  local expected="data/${seed}"
  if [[ -n "${DATA_ROOT:-}" ]]; then
    if _echohate_path_has_seed_bucket "${DATA_ROOT}" "${seed}"; then
      export DATA_ROOT
      return 0
    fi
    echo "[echohate_data_paths] NOTE: DATA_ROOT=${DATA_ROOT} -> ${expected} (ECHOHATE_DATA_SEED_DIR=${seed})" >&2
  fi
  DATA_ROOT="${expected}"
  export DATA_ROOT
}

# True iff path contains seed as a full path segment (not a prefix of another bucket).
# e.g. seed_0_model must NOT match .../seed_0_model_wCL/...
_echohate_path_has_seed_bucket() {
  local path="${1%/}"
  local seed="$2"
  [[ -z "${path}" || -z "${seed}" ]] && return 1
  [[ "${path}" == "${seed}" || "${path}" == "data/${seed}" ]] && return 0
  [[ "${path}" == *"/${seed}" || "${path}" == *"/${seed}/"* ]] && return 0
  [[ "${path}" == "${seed}/"* || "${path}" == "data/${seed}/"* ]] && return 0
  return 1
}

echohate_assert_seed_paths_consistent() {
  local out_dir="${1:-${OUTPUT_DIR:-}}"
  local seed="${ECHOHATE_DATA_SEED_DIR:-}"
  if [[ -z "${seed}" ]]; then
    return 0
  fi
  if [[ -n "${DATA_ROOT:-}" ]] && ! _echohate_path_has_seed_bucket "${DATA_ROOT}" "${seed}"; then
    echo "ERROR: DATA_ROOT=${DATA_ROOT} does not match ECHOHATE_DATA_SEED_DIR=${seed} (path-segment match; seed_0_model ≠ seed_0_model_wCL)" >&2
    return 1
  fi
  if [[ -n "${out_dir}" ]] && ! _echohate_path_has_seed_bucket "${out_dir}" "${seed}"; then
    echo "ERROR: OUTPUT_DIR=${out_dir} does not match ECHOHATE_DATA_SEED_DIR=${seed} (path-segment match; seed_0_model ≠ seed_0_model_wCL)" >&2
    return 1
  fi
  return 0
}
