#!/usr/bin/env bash
# Bootstrap data/seed_0_mech/ for 8-way mech-only encoder + LLM replace-around-τ.
#
# Copies from seed_0_comparator:
#   - prep: intermediate_data, splitted_data_80_10_10, clustered_811
#   - wCL student CSVs (training_dynamics, predictions, metrics) for var25
#
# Does NOT copy:
#   - checkpoints / model / tokenizer (weights stay on HPC)
#   - toy_prompt_* (re-infer with hard_prompt_GT_8way / nocontext_8way)
#   - C0 / I1 / woCL / pcon mix trees
#
# Stack: wCL encoder → var25 → 8-way mech-only aux → LLM replacement around τ.
#
# Usage (from experiment/echohate):
#   bash scripts/bootstrap_seed_0_mech.sh
#   FORCE=1 bash scripts/bootstrap_seed_0_mech.sh
#   DRY_RUN=1 bash scripts/bootstrap_seed_0_mech.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ECHOHATE_DATA_ROOT:-${ROOT}/data}"
FROM_DIR="${FROM_DIR:-seed_0_comparator}"
TO_DIR="${TO_DIR:-seed_0_mech}"
BASELINE="${BASELINE:-roberta_large_baseline_wCL}"

SRC="${DATA_ROOT}/${FROM_DIR}"
TGT="${DATA_ROOT}/${TO_DIR}"

if [[ ! -d "${SRC}" ]]; then
  echo "ERROR: source missing: ${SRC}" >&2
  exit 1
fi

if [[ -d "${TGT}" ]] && [[ "${FORCE:-}" != "1" ]]; then
  if find "${TGT}" -mindepth 2 -maxdepth 2 -type d -name 'splitted_data_80_10_10' 2>/dev/null | grep -q .; then
    echo "SKIP ${TO_DIR} (already bootstrapped; set FORCE=1 to refresh)" >&2
    exit 0
  fi
fi

echo "=== Bootstrap prep splits: ${FROM_DIR} -> ${TO_DIR} ==="
bash "${ROOT}/scripts/bootstrap_data_seed_dir.sh" \
  --from "${FROM_DIR}" \
  --to "${TO_DIR}" \
  --training-seed 0 \
  --data-root "${DATA_ROOT}"

_copy_wcl_csvs() {
  local src_run="$1" dst_run="$2"
  if [[ ! -d "${src_run}" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN:-}" == "1" ]]; then
    echo "DRY_RUN ${src_run} CSVs -> ${dst_run}"
    return 0
  fi
  mkdir -p "${dst_run}"
  local f
  for f in \
    training_dynamics.csv \
    training_dynamics_all_epochs.csv \
    metric_summary.json \
    metrics.json \
    run_hyperparams.txt \
    predictions_eval.csv \
    predictions_eval_tau050.csv \
    predictions_test.csv \
    predictions_test_tau050.csv \
    predictions_train.csv \
    predictions_train_tau050.csv
  do
    if [[ -f "${src_run}/${f}" ]]; then
      rsync -a "${src_run}/${f}" "${dst_run}/${f}"
    fi
  done
  if [[ -d "${src_run}/training_dynamics_epochs" ]]; then
    rsync -a "${src_run}/training_dynamics_epochs/" "${dst_run}/training_dynamics_epochs/"
  fi
}

echo "=== Copy ${BASELINE} dynamics/preds (no checkpoints, no toy_prompt) ==="
COPIED=()
while IFS= read -r ds; do
  [[ -d "${SRC}/${ds}" ]] || continue
  case "${ds}" in
    aggregate_*|.*) continue ;;
  esac
  src_run="${SRC}/${ds}/${BASELINE}"
  if [[ ! -d "${src_run}" ]]; then
    echo "  SKIP ${ds}: no ${BASELINE}"
    continue
  fi
  echo "  ${ds}/${BASELINE}"
  _copy_wcl_csvs "${src_run}" "${TGT}/${ds}/${BASELINE}"
  COPIED+=("${ds}")
done < <(find "${SRC}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)

MANIFEST="${TGT}/seed_manifest.json"
if [[ "${DRY_RUN:-}" != "1" ]]; then
  python3 - "${MANIFEST}" "${FROM_DIR}" "${TO_DIR}" "${BASELINE}" "$(IFS=,; echo "${COPIED[*]}")" <<'PY'
import json, sys
from datetime import datetime, timezone

path, source, target, baseline, copied = sys.argv[1:6]
datasets = [d for d in copied.split(",") if d]
doc = {
    "training_random_state": 0,
    "bootstrapped_from": source,
    "target_dir": target,
    "purpose": "8-way mech-only encoder (var25) then LLM replacement around τ",
    "student": baseline,
    "mechanism_taxonomy": "8-way hate (6 hate + Not Offensive + Offensive-no-PGA); SBIC stays 5-way offensive",
    "prompt_modes": ["nocontext_8way", "hard_prompt_GT_8way"],
    "compare_to": baseline,
    "split_note": "splitted_data_80_10_10 copied from source; split RNG seed 42",
    "prep_dirs": ["intermediate_data", "splitted_data_80_10_10", "clustered_811"],
    "baselines_copied": [baseline],
    "baselines_copied_datasets": datasets,
    "baselines_note": (
        f"Copied {baseline} training_dynamics + predictions/metrics only. "
        "No checkpoints, no toy_prompt_*, no C0/I1/woCL."
    ),
    "hard_case_selection": {
        "method": "variability_top25",
        "alias": "var25",
        "source_file": f"{baseline}/training_dynamics.csv",
        "subdir": "selection_variability_top25",
        "top_frac": 0.25,
        "baselines": [baseline],
    },
    "next_steps": [
        f"SEED_DIR={target} METHOD=var25 BASELINES={baseline} bash scripts/select_hard_cases_for_mech.sh",
        "LLM infer on var25 with --prompt-mode hard_prompt_GT_8way (phi + qwen)",
        "Train mech-only encoder (8-way head) on var25; pick by val hybrid / val conflict F1",
        "LLM replacement around τ (UGI band)",
    ],
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
open(path, "w", encoding="utf-8").write(json.dumps(doc, indent=2) + "\n")
PY
  echo "Wrote ${MANIFEST}"
fi

echo "Done. ${TO_DIR} ready under ${DATA_ROOT}/ (splits + ${BASELINE} CSVs)"
