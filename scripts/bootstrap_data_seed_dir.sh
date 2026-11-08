#!/usr/bin/env bash
# Bootstrap data/seed_<N>_model[_new|_wCL|_woCL]/ from an existing seed bucket.
#
# Always copies prep assets (not raw dumps — keep those on seed_0 only):
#   intermediate_data, splitted_data_80_10_10, clustered_811
# Does NOT copy: raw_data, preprocessed_data, encoder/, baseline/
#
# Optional --with-llm also copies (no encoder / no baseline weights):
#   llm_*  abstracted_*  (nocontext/withcontext teachers + abstracts)
#
# Training uses RANDOM_STATE=<N> later; splits stay identical to source (split seed 42).
#
# Usage (from experiment/echohate):
#   bash scripts/bootstrap_data_seed_dir.sh 44 123
#   bash scripts/bootstrap_data_seed_dir.sh --from seed_0_model 44
#   bash scripts/bootstrap_data_seed_dir.sh --from seed_0_model_new --suffix _model_new 1 42 777 1234
#   FORCE=1 bash scripts/bootstrap_data_seed_dir.sh --from seed_0_model_wCL --suffix _model_wCL --with-llm 1 42 777 1234
#   bash scripts/bootstrap_data_seed_dir.sh --from eacl_seed0 --prefix eacl_seed 42 1234
#   bash scripts/bootstrap_data_seed_dir.sh --from eacl_seed0 --to eacl_seed0_seq
#   DRY_RUN=1 bash scripts/bootstrap_data_seed_dir.sh 44
#
# Env:
#   SEED_DIR_SUFFIX   default _model (override with --suffix); ignored if --prefix is set
#   SEED_DIR_PREFIX   if set (e.g. eacl_seed), target is <prefix><N> instead of seed_<N><suffix>
#   FORCE=1           refresh prep (+llm) dirs even if target seed bucket exists
#   WITH_LLM=1        same as --with-llm

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ECHOHATE_DATA_ROOT:-${ROOT}/data}"
FROM_DIR="seed_0_model"
SUFFIX="${SEED_DIR_SUFFIX:-_model}"
PREFIX="${SEED_DIR_PREFIX:-}"
TO_DIR="${TO_DIR:-}"
TRAINING_SEED="${TRAINING_SEED:-}"
WITH_LLM="${WITH_LLM:-0}"
SEEDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      FROM_DIR="${2:?--from requires seed dir name like seed_0_model}"
      shift 2
      ;;
    --suffix)
      SUFFIX="${2:?--suffix requires value like _model or _model_new}"
      shift 2
      ;;
    --prefix)
      PREFIX="${2:?--prefix requires value like eacl_seed}"
      shift 2
      ;;
    --to)
      TO_DIR="${2:?--to requires target bucket name like eacl_seed0_seq}"
      shift 2
      ;;
    --training-seed)
      TRAINING_SEED="${2:?--training-seed requires integer}"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="${2:?--data-root requires path}"
      shift 2
      ;;
    --with-llm)
      WITH_LLM=1
      shift
      ;;
    -h|--help)
      sed -n '2,35p' "$0"
      exit 0
      ;;
    *)
      SEEDS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${TO_DIR}" && ${#SEEDS[@]} -gt 0 ]]; then
  echo "ERROR: use either --to <name> or numeric seed args, not both" >&2
  exit 1
fi
if [[ -z "${TO_DIR}" && ${#SEEDS[@]} -lt 1 ]]; then
  echo "Usage: $0 [--from seed_0_model] [--suffix _model|_model_new|_model_wCL] [--prefix eacl_seed] [--to eacl_seed0_seq] [--with-llm] <seed_number> ..." >&2
  echo "Example: FORCE=1 $0 --from seed_0_model_wCL --suffix _model_wCL --with-llm 1 42 777 1234" >&2
  echo "Example: $0 --from eacl_seed0 --prefix eacl_seed 42 1234" >&2
  echo "Example: $0 --from eacl_seed0 --to eacl_seed0_seq" >&2
  exit 1
fi
if [[ -n "${TO_DIR}" ]]; then
  SEEDS=(0)
fi

if [[ -z "${PREFIX}" && "${SUFFIX}" != _* ]]; then
  SUFFIX="_${SUFFIX}"
fi

SRC="${DATA_ROOT}/${FROM_DIR}"
if [[ ! -d "${SRC}" ]]; then
  echo "ERROR: source missing: ${SRC}" >&2
  exit 1
fi

PREP_DIRS=(intermediate_data splitted_data_80_10_10 clustered_811)
# raw_data / preprocessed_data stay on seed_0 only — later seeds reuse splits+teachers, not raw dumps.

DATASETS=()
while IFS= read -r _ds; do
  DATASETS+=("${_ds}")
done < <(find "${SRC}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)

_copy_tree() {
  local src="$1" dst="$2"
  if [[ "${DRY_RUN:-}" == "1" ]]; then
    echo "DRY_RUN rsync ${src}/ -> ${dst}/"
    return 0
  fi
  mkdir -p "${dst}"
  rsync -a "${src}/" "${dst}/"
}

for seed in "${SEEDS[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: seed must be numeric, got: ${seed}" >&2
    exit 1
  fi
  if [[ -n "${TO_DIR}" ]]; then
    TGT_NAME="${TO_DIR}"
    train_seed="${TRAINING_SEED:-${seed}}"
  elif [[ -n "${PREFIX}" ]]; then
    TGT_NAME="${PREFIX}${seed}"
    train_seed="${TRAINING_SEED:-${seed}}"
  else
    TGT_NAME="seed_${seed}${SUFFIX}"
    train_seed="${TRAINING_SEED:-${seed}}"
  fi
  TGT="${DATA_ROOT}/${TGT_NAME}"
  if [[ -d "${TGT}" ]] && [[ "${FORCE:-}" != "1" ]]; then
    # Only skip if a real prep asset already exists (placeholder DS dirs are OK to fill).
    if find "${TGT}" -mindepth 2 -maxdepth 2 -type d \( \
        -name 'splitted_data_80_10_10' -o -name 'raw_data' -o -name 'llm_*' \
      \) 2>/dev/null | grep -q .; then
      echo "SKIP ${TGT_NAME} (has prep/llm dirs; set FORCE=1 to refresh)" >&2
      continue
    fi
    echo "NOTE ${TGT_NAME} exists as placeholder; bootstrapping into it"
  fi
  echo "=== Bootstrap ${TGT_NAME} from ${FROM_DIR} (training seed ${train_seed}; with_llm=${WITH_LLM}) ==="
  mkdir -p "${TGT}"
  for ds in "${DATASETS[@]}"; do
    [[ -d "${SRC}/${ds}" ]] || continue
    # Skip non-dataset dirs (aggregate_*, seed_manifest parent handled separately).
    case "${ds}" in
      aggregate_*|.*) continue ;;
    esac
    for sub in "${PREP_DIRS[@]}"; do
      if [[ -d "${SRC}/${ds}/${sub}" ]]; then
        echo "  ${ds}/${sub}"
        _copy_tree "${SRC}/${ds}/${sub}" "${TGT}/${ds}/${sub}"
      fi
    done
    if [[ "${WITH_LLM}" == "1" ]]; then
      # Teachers + abstracts used by cue/hybrid; exclude heavy encoder/baseline trees.
      while IFS= read -r -d '' src_sub; do
        bn="$(basename "${src_sub}")"
        echo "  ${ds}/${bn}"
        _copy_tree "${src_sub}" "${TGT}/${ds}/${bn}"
      done < <(find "${SRC}/${ds}" -mindepth 1 -maxdepth 1 -type d \
        \( -name 'llm_*' -o -name 'abstracted_*' \) -print0 2>/dev/null)
    fi
  done
  MANIFEST="${TGT}/seed_manifest.json"
  if [[ "${DRY_RUN:-}" != "1" ]]; then
    python3 - "${MANIFEST}" "${train_seed}" "${FROM_DIR}" "${WITH_LLM}" "${SUFFIX}" "${TGT_NAME}" <<'PY'
import json, sys
from datetime import datetime, timezone
path, train_seed, source, with_llm, suffix, target_dir = (
    sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
)
cl = None
if suffix.endswith("_woCL"):
    cl = "ce_only"
elif suffix.endswith("_wCL"):
    cl = "with_contrastive_loss"
doc = {
    "training_random_state": train_seed,
    "bootstrapped_from": source,
    "target_dir": target_dir,
    "contrastive_loss": cl,
    "with_llm_copied": with_llm == "1",
    "split_note": "splitted_data_80_10_10 copied from source; split RNG seed 42 unless regenerated",
    "prep_dirs": [
        "intermediate_data",
        "splitted_data_80_10_10",
        "clustered_811",
    ],
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
open(path, "w", encoding="utf-8").write(json.dumps(doc, indent=2) + "\n")
PY
    echo "Wrote ${MANIFEST}"
  fi
  echo "Done ${TGT_NAME}"
done

echo "All requested seed dirs ready under ${DATA_ROOT}/"
