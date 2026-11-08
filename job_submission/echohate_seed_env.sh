#!/usr/bin/env bash
# shellcheck shell=bash
# Read seeds from config/echohate_project_config.json (via pipeline.common.project_config) and export canonical seed variables.
# Source from any script after optionally setting ECHOHATE_ROOT (defaults to parent of job_submission/).
#
# Exports (when unset): RANDOM_STATE, SEED, ECHOHATE_* stage seeds
# Always exports: ECHOHATE_CANONICAL_RANDOM_STATE, ECHOHATE_SEEDS_COMMA, ECHOHATE_SEEDS_SPACE
# Prefer sourcing echohate_project_env.sh for split/data/training from JSON as well.
#
# Usage:
#   # shellcheck source=/dev/null
#   source "${ECHOHATE_ROOT}/job_submission/echohate_seed_env.sh"

_rag_seed_this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ECHOHATE_ROOT="${ECHOHATE_ROOT:-$(cd "${_rag_seed_this_dir}/.." && pwd)}"

_echohate_seed_query() {
  local _op="$1"
  ECHOHATE_ROOT="${ECHOHATE_ROOT}" python3 -c "
import os, sys
root = os.environ['ECHOHATE_ROOT']
sys.path.insert(0, root)
from pipeline.common.project_config import (
    default_random_state,
    default_seed_sweep,
    default_split_random_state,
)
op = sys.argv[1]
r = int(default_random_state())
s = [int(x) for x in default_seed_sweep()]
if op == 'rs':
    print(r)
elif op == 'split':
    print(int(default_split_random_state()))
elif op == 'comma':
    print(','.join(map(str, s)))
elif op == 'space':
    print(' '.join(map(str, s)))
else:
    raise SystemExit('bad op')
" "${_op}"
}

if ECHOHATE_CANONICAL_RANDOM_STATE="$(_echohate_seed_query rs 2>/dev/null)"; then
  export ECHOHATE_CANONICAL_RANDOM_STATE
  export ECHOHATE_SEEDS_COMMA="$(_echohate_seed_query comma)"
  export ECHOHATE_SEEDS_SPACE="$(_echohate_seed_query space)"
else
  echo "[echohate_seed_env] WARN: python seed query failed; using fallbacks" >&2
  export ECHOHATE_CANONICAL_RANDOM_STATE="${ECHOHATE_CANONICAL_RANDOM_STATE:-0}"
  export ECHOHATE_SEEDS_COMMA="${ECHOHATE_SEEDS_COMMA:-0}"
  export ECHOHATE_SEEDS_SPACE="${ECHOHATE_SEEDS_SPACE:-0}"
fi

: "${RANDOM_STATE:=${ECHOHATE_CANONICAL_RANDOM_STATE}}"
: "${SEED:=${ECHOHATE_CANONICAL_RANDOM_STATE}}"
: "${ECHOHATE_SPLIT_SEED:=$(_echohate_seed_query split 2>/dev/null || echo 42)}"
export ECHOHATE_SPLIT_SEED
