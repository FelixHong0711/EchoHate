#!/usr/bin/env bash
# shellcheck shell=bash
# Load HPC_PASSWORD from secrets/hpc_password (gitignored) for sshpass automation.
#
#   echohate_export_hpc_password_from_secrets [ECHOHATE_ROOT]
#
# Precedence: existing HPC_PASSWORD or SSHPASS in the environment wins.
# Requires: sshpass on PATH (Mac: brew install hudochenkov/sshpass/sshpass)

echohate_export_hpc_password_from_secrets() {
  local root="${1:-${ECHOHATE_ROOT:-${LOCAL_ECHOHATE:-}}}"
  if [ -n "${HPC_PASSWORD:-}" ] || [ -n "${SSHPASS:-}" ]; then
    if [ -n "${HPC_PASSWORD:-}" ] && [ -z "${SSHPASS:-}" ]; then
      export SSHPASS="${HPC_PASSWORD}"
    elif [ -n "${SSHPASS:-}" ] && [ -z "${HPC_PASSWORD:-}" ]; then
      export HPC_PASSWORD="${SSHPASS}"
    fi
    return 0
  fi
  if [ -z "${root}" ] || [ ! -d "${root}" ]; then
    return 0
  fi
  local f="${root}/secrets/hpc_password"
  if [ ! -f "${f}" ]; then
    return 0
  fi
  HPC_PASSWORD="$(tr -d ' \t\r\n' < "${f}")"
  if [ -z "${HPC_PASSWORD}" ]; then
    return 0
  fi
  export HPC_PASSWORD
  export SSHPASS="${HPC_PASSWORD}"
}
