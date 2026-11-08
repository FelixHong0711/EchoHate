#!/usr/bin/env bash
# Source from Qwen submit wrappers — stale Phi exports in the login shell must not win via ${VAR:-default}.
# Usage (inside a wrapper, after set -euo pipefail):
#   # shellcheck source=/dev/null
#   source "${_BOOT_DIR}/_force_qwen_teacher_env.sh"

unset TEACHER_LM_FAMILY TEACHER_OUTPUT_TAG TEACHER_OUTPUT_TAG_OVERRIDES \
  TEACHER_OUTPUT_TAG_PHI TEACHER_OUTPUT_TAG_OVERRIDES_PHI \
  TEACHER_OUTPUT_TAG_LLAMA TEACHER_OUTPUT_TAG_OVERRIDES_LLAMA 2>/dev/null || true

export TEACHER_LM_FAMILY=qwen
export TEACHER_OUTPUT_TAG_QWEN="${TEACHER_OUTPUT_TAG_QWEN:-_new_prompt}"
export TEACHER_OUTPUT_TAG="${TEACHER_OUTPUT_TAG_QWEN}"
