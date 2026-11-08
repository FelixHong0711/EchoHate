#!/usr/bin/env bash
# shellcheck shell=bash
# Per-dataset prompt task (hate | offensive). Mirrors config/echohate_project_config.json dataset_tasks.
#
# Usage:
#   source "${ECHOHATE_ROOT}/job_submission/echohate_dataset_tasks.sh"
#   echohate_prompt_task_for_dataset SBIC   # -> offensive
#   echohate_offensive_buckets_for_dataset SBIC  # -> 1

echohate_prompt_task_for_dataset() {
  # Always resolve from dataset name. Do NOT inherit ambient PROMPT_TASK here —
  # batch submit scripts call this per-dataset; inheriting would leak the previous
  # dataset's task (e.g. hate → SBIC). Override explicitly via PROMPT_TASK=... when
  # calling submit scripts, or set after this helper returns.
  local ds="${1:-}"
  if [[ -n "${ABSTRACT_TASK:-}" && "${ECHOHATE_INHERIT_ABSTRACT_TASK:-0}" == "1" ]]; then
    echo "${ABSTRACT_TASK}"
    return 0
  fi
  case "${ds}" in
    SBIC) echo "offensive" ;;
    *) echo "hate" ;;
  esac
}

echohate_offensive_buckets_for_dataset() {
  local ds="${1:-}"
  if [[ -n "${OFFENSIVE_BUCKETS:-}" && "${OFFENSIVE_BUCKETS}" != "0" ]]; then
    echo "1"
    return 0
  fi
  case "${ds}" in
    SBIC) echo "1" ;;
    *) echo "0" ;;
  esac
}

echohate_apply_dataset_task_defaults() {
  local ds="${1:-${DATASET:-}}"
  [[ -n "${ds}" ]] || return 0
  if [[ -z "${ABSTRACT_TASK:-}" ]]; then
    ABSTRACT_TASK="$(echohate_prompt_task_for_dataset "${ds}")"
    export ABSTRACT_TASK
  fi
  if [[ -z "${OFFENSIVE_BUCKETS:-}" ]]; then
    OFFENSIVE_BUCKETS="$(echohate_offensive_buckets_for_dataset "${ds}")"
    export OFFENSIVE_BUCKETS
  fi
  if [[ -z "${PROMPT_TASK:-}" ]]; then
    PROMPT_TASK="$(echohate_prompt_task_for_dataset "${ds}")"
    export PROMPT_TASK
  fi
}
