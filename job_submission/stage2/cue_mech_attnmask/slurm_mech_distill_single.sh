#!/usr/bin/env bash
#SBATCH --job-name=cue_dist_mech
#SBATCH --output=cue_dist_mech_%j.out
#SBATCH --error=cue_dist_mech_%j.err
# Partition: only via sbatch CLI from submit_roberta_fft_mech_cue_post_llm_from_local.sh (gpu vs dgx).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#
# Modes:
#   Single λ (default): LAMBDA_MECH + OUTPUT_DIR
#   Packed λ grid:      LAMBDA_MECH_LIST=0.03+0.05+… (plus-sep; commas break Slurm --export)
#                       MECH_OUT_DIR_PREFIX + MECH_OUT_DIR_SUFFIX
#                       → OUTPUT_DIR=${PREFIX}lm${lm_tag}${SUFFIX} for each λ
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

echo "=========================================="
echo "encoder distill mech job"
echo "job_id=${SLURM_JOB_ID:-na} node=${SLURM_NODELIST:-na}"
echo "dataset=${DATASET:-na} model=${MODEL_NAME:-na}"
echo "output_dir=${OUTPUT_DIR:-na}"
echo "LAMBDA_MECH_LIST=${LAMBDA_MECH_LIST:-<single>}"
echo "MECH_OUT_DIR_PREFIX=${MECH_OUT_DIR_PREFIX:-na}"
echo "MECH_OUT_DIR_SUFFIX=${MECH_OUT_DIR_SUFFIX:-na}"
echo "RUN_FINAL_TEST_EVAL=${RUN_FINAL_TEST_EVAL:-1} RUN_EVAL_OVERWRITE_OUTPUTS=${RUN_EVAL_OVERWRITE_OUTPUTS:-0}"
echo "FORCE_RETRAIN=${FORCE_RETRAIN:-0}"
echo "LAMBDA_MECH=${LAMBDA_MECH:-na}"
echo "MECH_AUX_MODE=${MECH_AUX_MODE:-mechanism}"
echo "HATE_MECH_TAXONOMY=${HATE_MECH_TAXONOMY:-8way}"
echo "=========================================="

export HPC_REQUIRE_CUDA="${HPC_REQUIRE_CUDA:-1}"
export HPC_MODULE_PURGE="${HPC_MODULE_PURGE:-0}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

ECHOHATE_ROOT="${SLURM_SUBMIT_DIR:-.}"
cd "${ECHOHATE_ROOT}"
# shellcheck source=/dev/null
source "${ECHOHATE_ROOT}/job_submission/training_hyper_defaults.sh"

set +u
# shellcheck source=/dev/null
source "${ECHOHATE_ROOT}/job_submission/hpc_env_deakin.sh"
set -u

export HATE_MECH_TAXONOMY=8way
export ALLOW_TAXONOMY_OVERWRITE="${ALLOW_TAXONOMY_OVERWRITE:-0}"

export ECHOHATE_SPLITTED_DATA_SUBDIR="${ECHOHATE_SPLITTED_DATA_SUBDIR:-}"

ACCEPTED_K_ARGS=()
if [ -n "${ACCEPTED_SELECTED_K_CSV:-}" ]; then
  ACCEPTED_K_ARGS+=(--accepted_selected_k_csv "${ACCEPTED_SELECTED_K_CSV}")
fi

echo "Preflight checks (mech)..."
export LLM_FILE="${LLM_FILE:-}"
export ACCEPTED_SELECTED_K_CSV="${ACCEPTED_SELECTED_K_CSV:-}"
export DATA_ROOT="${DATA_ROOT:-${ECHOHATE_ROOT}/data}"
export ECHOHATE_SPLITTED_DATA_SUBDIR="${ECHOHATE_SPLITTED_DATA_SUBDIR:-}"
export DATASET
python - <<'PY'
import os
import pandas as pd

llm = os.environ.get("LLM_FILE", "").strip()
accepted = os.environ.get("ACCEPTED_SELECTED_K_CSV", "").strip()
data_root = os.environ.get("DATA_ROOT") or os.path.join(os.environ.get("ECHOHATE_ROOT", "."), "data")
dataset = os.environ["DATASET"]
split_sub = os.environ.get("ECHOHATE_SPLITTED_DATA_SUBDIR", "")
train_csv = os.path.join(data_root, dataset, split_sub, "train.csv")

print(f"  llm_file={llm or '<none>'}")
print(f"  accepted_selected_k_csv={accepted or '<none>'}")
print(f"  train_csv={train_csv}")
if not os.path.isfile(train_csv):
    raise SystemExit(f"ERROR: missing train split: {train_csv}")
if not llm and not accepted:
    raise SystemExit("ERROR: set LLM_FILE or ACCEPTED_SELECTED_K_CSV")

tr = pd.read_csv(train_csv)
train_ids = {f"train_{i}" for i in range(len(tr))}

def _check(path, need):
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: missing mechanism file: {path}")
    table = pd.read_csv(path)
    missing = sorted(c for c in need if c not in table.columns)
    if missing:
        raise SystemExit(f"ERROR: {path} missing columns: {missing}")
    prefix = table["row_id"].astype(str).str.extract(r"^(\w+)_", expand=False).fillna("none")
    print("  row_id_prefix_counts=", prefix.value_counts().to_dict())
    hit = int(table["row_id"].astype(str).isin(train_ids).sum())
    print(f"  train_id_hits={hit}/{len(table)} file={path}")
    print(f"  rows={len(table)} train_rows={len(tr)}")

if llm:
    _check(llm, {"row_id", "mechanism"})
if accepted:
    _check(accepted, {"row_id", "mechanism"})
PY

_mech_training_artifacts_ok() {
  local d="$1"
  [ -s "${d}/encoder_model.pt" ] || return 1
  return 0
}

_mech_checkpoint_can_skip_train() {
  local d="$1"
  _mech_training_artifacts_ok "${d}" || return 1
  python -m pipeline.stage2.check_mech_checkpoint --run-dir "${d}" --approach mech --require-artifacts
}

_mech_refuse_taxonomy_overwrite() {
  local d="$1"
  local cfg=""
  for _c in "${d}/encoder_distill_config.json" "${d}/metric_summary.json"; do
    [[ -s "${_c}" ]] && cfg="${_c}" && break
  done
  [[ -n "${cfg}" ]] || return 0
  HATE_MECH_TAXONOMY="${HATE_MECH_TAXONOMY:-8way}" \
  MECH_AUX_MODE="${MECH_AUX_MODE:-mechanism}" \
  ALLOW_TAXONOMY_OVERWRITE="${ALLOW_TAXONOMY_OVERWRITE:-0}" \
    python - "${cfg}" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1]))
saved = cfg.get("n_mechanism_classes")
if saved is None and isinstance(cfg.get("training_config"), dict):
    saved = cfg["training_config"].get("n_mechanism_classes")
if saved is None:
    raise SystemExit(0)
saved = int(saved)
aux = (
    str(cfg.get("mech_aux_mode") or os.environ.get("MECH_AUX_MODE") or "mechanism")
    .strip()
    .lower()
)
if aux == "binary":
    expect = 2
elif os.environ.get("DATASET", "").strip().upper() == "SBIC":
    expect = 5
else:
    expect = 8
if saved == expect:
    raise SystemExit(0)
if os.environ.get("ALLOW_TAXONOMY_OVERWRITE", "0") == "1":
    print(f"ALLOW_TAXONOMY_OVERWRITE=1: replacing n_mechanism_classes={saved} with expect={expect}")
    raise SystemExit(0)
raise SystemExit(
    f"ERROR: refusing overwrite of {sys.argv[1]} (saved n_mechanism_classes={saved}, "
    f"expect={expect} aux={aux}). Keep existing dirs; use a new ENCODER_RUN_TAG "
    f"for W2 controls, or set ALLOW_TAXONOMY_OVERWRITE=1 intentionally."
)
PY
}

_resolve_ce_cl_from_paths() {
  # Resolve CE↔CL from OUTPUT_DIR / DATA_ROOT / packed prefix (mutates CE_CONTRASTIVE_LAMBDA).
  local _p_cl="${OUTPUT_DIR:-}:${MECH_OUT_DIR_PREFIX:-}:${DATA_ROOT:-}"
  case "${_p_cl}" in
    *_woCL*)
      if [[ -n "${CE_CONTRASTIVE_LAMBDA:-}" && "${CE_CONTRASTIVE_LAMBDA}" != "none" && "${CE_CONTRASTIVE_LAMBDA}" != "off" ]]; then
        echo "WARN: woCL path but CE_CONTRASTIVE_LAMBDA='${CE_CONTRASTIVE_LAMBDA}' — forcing CE-only"
      fi
      CE_CONTRASTIVE_LAMBDA=""
      ;;
    *_wCL*)
      if [[ -z "${CE_CONTRASTIVE_LAMBDA:-}" || "${CE_CONTRASTIVE_LAMBDA}" == "none" || "${CE_CONTRASTIVE_LAMBDA}" == "off" ]]; then
        echo "WARN: wCL path but CE_CONTRASTIVE_LAMBDA empty/none — forcing 0.5"
        CE_CONTRASTIVE_LAMBDA=0.5
      fi
      ;;
  esac
  if [[ "${CE_CONTRASTIVE_LAMBDA:-}" == "none" || "${CE_CONTRASTIVE_LAMBDA:-}" == "off" ]]; then
    CE_CONTRASTIVE_LAMBDA=""
  fi
  if [[ "${_p_cl}" == *"_woCL"* && -n "${CE_CONTRASTIVE_LAMBDA:-}" ]]; then
    echo "ERROR: path is woCL but CE_CONTRASTIVE_LAMBDA='${CE_CONTRASTIVE_LAMBDA}'" >&2
    exit 1
  fi
  if [[ "${_p_cl}" == *"_wCL"* && "${_p_cl}" != *"_woCL"* && -z "${CE_CONTRASTIVE_LAMBDA:-}" ]]; then
    echo "ERROR: path is wCL but CE_CONTRASTIVE_LAMBDA is empty" >&2
    exit 1
  fi
  echo "CE_CONTRASTIVE_LAMBDA=${CE_CONTRASTIVE_LAMBDA:-(CE only)}  CONTRASTIVE_MARGIN=${CONTRASTIVE_MARGIN:-0.75}"
}

_refuse_warmft() {
  if [[ -n "${INIT_BACKBONE_FROM:-}" ]] || [[ "${INIT_BACKBONE_AUTO:-0}" == "1" ]]; then
    echo "ERROR: warmft / INIT_BACKBONE_FROM not supported; train from scratch." >&2
    exit 1
  fi
  if [[ "${OUTPUT_DIR:-}" == *warmft* || "${MECH_OUT_DIR_PREFIX:-}" == *warmft* || "${MECH_OUT_DIR_SUFFIX:-}" == *warmft* ]]; then
    echo "ERROR: warmft path tag not supported — use from-scratch dirs." >&2
    exit 1
  fi
}

_run_one_mech_cell() {
  local OUTPUT_DIR="$1"
  local LAMBDA_MECH="$2"
  export OUTPUT_DIR LAMBDA_MECH

  echo ""
  echo "=========================================="
  echo "MECH cell λ=${LAMBDA_MECH}  out=${OUTPUT_DIR}"
  echo "=========================================="

  FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

  if [[ "${FORCE_RETRAIN}" != "1" ]] && _mech_checkpoint_can_skip_train "${OUTPUT_DIR}"; then
    echo "encoder_model.pt present and mechanism-head compatible under ${OUTPUT_DIR}; skipping train (FORCE_RETRAIN=1 to retrain)."
  else
    if [[ "${FORCE_RETRAIN}" != "1" ]] && _mech_training_artifacts_ok "${OUTPUT_DIR}"; then
      if ! _mech_refuse_taxonomy_overwrite "${OUTPUT_DIR}"; then
        exit 1
      fi
      echo "WARNING: stale mechanism-head checkpoint under ${OUTPUT_DIR}; retraining automatically."
    elif [[ "${FORCE_RETRAIN}" == "1" ]] && _mech_training_artifacts_ok "${OUTPUT_DIR}"; then
      _mech_refuse_taxonomy_overwrite "${OUTPUT_DIR}" || exit 1
    fi
    CL_FLAGS=()
    if [[ -n "${CE_CONTRASTIVE_LAMBDA:-}" ]]; then
      CL_FLAGS+=(--ce_contrastive_lambda "${CE_CONTRASTIVE_LAMBDA}")
    fi
    if [[ -n "${CONTRASTIVE_MARGIN:-}" ]]; then
      CL_FLAGS+=(--contrastive_margin "${CONTRASTIVE_MARGIN}")
    fi
    echo "CL_FLAGS=${#CL_FLAGS[@]} (train path)"
    DYNAMICS_FLAGS=()
    if [[ "${TRACK_DYNAMICS:-1}" == "1" ]]; then
      DYNAMICS_FLAGS+=(--track-dynamics)
      echo "TRACK_DYNAMICS=1 → --track-dynamics"
    fi
    LLM_ARGS=()
    if [[ -n "${LLM_FILE:-}" ]]; then
      LLM_ARGS+=(--llm_file "${LLM_FILE}")
    fi
    python -m pipeline.stage2.mech_distill \
      --dataset "${DATASET}" \
      --data_root "${DATA_ROOT:-${ECHOHATE_ROOT}/data}" \
      "${LLM_ARGS[@]}" \
      --evaluation_split "${EVALUATION_SPLIT:-validation}" \
      "${ACCEPTED_K_ARGS[@]}" \
      --output_dir "${OUTPUT_DIR}" \
      --model_name "${MODEL_NAME}" \
      --num_epochs "${NUM_EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --eval_batch_size "${EVAL_BATCH_SIZE}" \
      --learning_rate "${LEARNING_RATE}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --warmup_ratio "${WARMUP_RATIO}" \
      --max_length "${MAX_LENGTH}" \
      --random_state "${RANDOM_STATE}" \
      --pos_weight_mode "${POS_WEIGHT_MODE}" \
      --class_weight_target "${CLASS_WEIGHT_TARGET}" \
      --lambda_mech "${LAMBDA_MECH}" \
      --lambda_mech_ramp_ratio "${LAMBDA_MECH_RAMP_RATIO:-0.1}" \
      --mech_aux_mode "${MECH_AUX_MODE:-mechanism}" \
      "${CL_FLAGS[@]}" \
      "${DYNAMICS_FLAGS[@]}" \
      --require_gpu
  fi

  RUN_FINAL_TEST_EVAL="${RUN_FINAL_TEST_EVAL:-1}"
  if [ "${RUN_FINAL_TEST_EVAL}" = "1" ] || [ "${RUN_FINAL_TEST_EVAL}" = "true" ]; then
    echo "Running final test evaluation (mech) from saved checkpoint..."
    EVAL_EXTRA=()
    if [ -n "${ECHOHATE_SPLITTED_DATA_SUBDIR:-}" ]; then
      EVAL_EXTRA+=(--splitted_data_subdir "${ECHOHATE_SPLITTED_DATA_SUBDIR}")
    fi
    if [ "${RUN_EVAL_OVERWRITE_OUTPUTS:-0}" = "1" ] || [ "${RUN_EVAL_OVERWRITE_OUTPUTS:-}" = "true" ]; then
      EVAL_EXTRA+=(--overwrite_outputs)
    fi
    python -m pipeline.stage2.run_eval_test_from_checkpoint \
      --approach mech \
      --dataset "${DATASET}" \
      --data_root "${DATA_ROOT:-${ECHOHATE_ROOT}/data}" \
      --run_dir "${OUTPUT_DIR}" \
      --batch_size "${EVAL_BATCH_SIZE}" \
      --tune_split validation \
      --output_prefix "final_test" \
      "${EVAL_EXTRA[@]}"
  fi
}

_resolve_ce_cl_from_paths
_refuse_warmft

# Packed grid: plus-separated λ list (commas break Slurm --export).
if [[ -n "${LAMBDA_MECH_LIST:-}" ]]; then
  if [[ -z "${MECH_OUT_DIR_PREFIX:-}" ]]; then
    echo "ERROR: LAMBDA_MECH_LIST set but MECH_OUT_DIR_PREFIX is empty" >&2
    exit 1
  fi
  MECH_OUT_DIR_SUFFIX="${MECH_OUT_DIR_SUFFIX:-}"
  IFS='+' read -ra _LM_ARR <<< "${LAMBDA_MECH_LIST}"
  echo "Packed λ grid: n=${#_LM_ARR[@]} list=${LAMBDA_MECH_LIST}"
  _cell_i=0
  for _lm in "${_LM_ARR[@]}"; do
    _lm="$(echo "${_lm}" | xargs)"
    [[ -z "${_lm}" ]] && continue
    _cell_i=$((_cell_i + 1))
    _lm_tag="${_lm/./p}"
    _out="${MECH_OUT_DIR_PREFIX}lm${_lm_tag}${MECH_OUT_DIR_SUFFIX}"
    echo "[pack ${_cell_i}/${#_LM_ARR[@]}] λ=${_lm} → ${_out}"
    _run_one_mech_cell "${_out}" "${_lm}"
  done
  echo "Packed λ grid done (n=${_cell_i})."
  exit 0
fi

# Single-λ path (default).
if [[ -z "${OUTPUT_DIR:-}" || -z "${LAMBDA_MECH:-}" ]]; then
  echo "ERROR: need OUTPUT_DIR+LAMBDA_MECH (single) or LAMBDA_MECH_LIST+MECH_OUT_DIR_PREFIX (pack)" >&2
  exit 1
fi
_run_one_mech_cell "${OUTPUT_DIR}" "${LAMBDA_MECH}"
