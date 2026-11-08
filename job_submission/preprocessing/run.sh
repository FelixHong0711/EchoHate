#!/usr/bin/env bash
# Preprocessing: stratified splits and dataset-specific ingest.
#
#   bash job_submission/preprocessing/run.sh dynahate --data-root data/seed_0 --dataset DynaHate
#   bash job_submission/preprocessing/run.sh split --datasets DynaHate --seed 42
#   bash job_submission/preprocessing/run.sh ihc
#   bash job_submission/preprocessing/run.sh sbic --data-root data/seed_0_model
#   bash job_submission/preprocessing/run.sh help

set -euo pipefail

_ECHOHATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${_ECHOHATE_ROOT}"

_cmd="${1:-help}"
shift || true

case "${_cmd}" in
  help|-h|--help)
    exec python -m pipeline.preprocessing --help
    ;;
  *)
    exec python -m pipeline.preprocessing "${_cmd}" "$@"
    ;;
esac
