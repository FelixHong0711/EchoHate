#!/usr/bin/env bash
# Submit RoBERTa FFT baseline to HPC (one dataset or all with splits under data/<seed>/).
#
# Single dataset:
#   DATASET=IHC MODEL_NAME=roberta-large bash job_submission/stage2/roberta_baseline/submit_roberta_baseline_from_local.sh
#
# All datasets in seed_0_model (default list below):
#   ECHOHATE_DATA_SEED_DIR=seed_0_model ALL_DATASETS=1 bash job_submission/stage2/roberta_baseline/submit_roberta_baseline_from_local.sh
# Conda on HPC (default rag_v8_gpu): CONDA_ENV=rag_v8_gpu ...
#
# Custom list:
#   DATASETS=IHC,DynaHate,Ethos bash job_submission/stage2/roberta_baseline/submit_roberta_baseline_from_local.sh
#
# Train on train+validation — 4 protocols (epochs × applied τ):
#   thr=0.5  6ep        → .../roberta_large_baseline_trainval/
#   thr=0.5  best-epoch → .../roberta_large_baseline_trainval_bestepoch/
#   thr=from80 6ep      → .../roberta_large_baseline_trainval_from80tau/
#   thr=from80 best-ep  → .../roberta_large_baseline_trainval_bestepoch_from80tau/
#
#   TRAIN_ON_TRAIN_VAL=1 TRAINVAL_APPLIED_THRESHOLD=0.5 DATASETS=... bash .../submit_roberta_baseline_from_local.sh
#   TRAIN_ON_TRAIN_VAL=1 TRAINVAL_APPLIED_THRESHOLD=0.5 EPOCHS_FROM_TRAIN_BASELINE=1 DATASETS=... bash ...
#   TRAIN_ON_TRAIN_VAL=1 TRAINVAL_APPLIED_THRESHOLD=from80 DATASETS=... bash ...
#   TRAIN_ON_TRAIN_VAL=1 TRAINVAL_APPLIED_THRESHOLD=from80 EPOCHS_FROM_TRAIN_BASELINE=1 DATASETS=... bash ...

set -euo pipefail

_BOOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_BOOT}/../../hpc_defaults_from_local.sh"
hpc_apply_ra_default_or_resolve
# Defaults for seed_0_model pipeline (JSON: split/abstract-cluster=42; GMM/abstract/model=0).
export ECHOHATE_DATA_SEED_DIR="${ECHOHATE_DATA_SEED_DIR:-seed_0_model}"
export RANDOM_STATE="${RANDOM_STATE:-0}"
# shellcheck source=/dev/null
source "${_BOOT}/../../echohate_project_env.sh"

HPC_DIR="${HPC_RA}/experiment/echohate"
CONDA_ENV="${CONDA_ENV:-rag_v8_gpu}"
CREATE_ENV="${CREATE_ENV:-0}"
# Prefer ROBERTA_MODEL_NAME. Ignore ambient MODEL_NAME unless it looks like an encoder
# (leftover Qwen/Phi exports otherwise full-FT a 14B and OOM on DGX).
if [[ -n "${ROBERTA_MODEL_NAME:-}" ]]; then
  MODEL_NAME="${ROBERTA_MODEL_NAME}"
elif [[ -n "${MODEL_NAME:-}" ]] && [[ "${MODEL_NAME}" == *[Rr]oberta* || "${MODEL_NAME}" == *[Bb]ert* || "${MODEL_NAME}" == *[Dd]eberta* ]]; then
  :
else
  if [[ -n "${MODEL_NAME:-}" ]]; then
    echo "WARNING: ignoring ambient MODEL_NAME=${MODEL_NAME} for roberta_baseline; using roberta-large (set ROBERTA_MODEL_NAME=... to override)." >&2
  fi
  MODEL_NAME="roberta-large"
fi
ECHOHATE_ROBERTA_BASELINE_DIR="${ECHOHATE_ROBERTA_BASELINE_DIR:-roberta_large_baseline}"
ECHOHATE_SPLITTED_DATA_SUBDIR="${ECHOHATE_SPLITTED_DATA_SUBDIR:-splitted_data_80_10_10}"
SLURM_SCRIPT="job_submission/stage2/roberta_baseline/slurm_roberta_baseline.sh"
# hpclogin_dgx → partition dgx + DGX nodelist (see hpc_defaults_from_local.sh).
# Drop stale SBATCH_EXTRA from the login shell — ${SBATCH_EXTRA:-...} would keep an old --nodelist=b-1,b-2.
if [[ -z "${SBATCH_EXTRA_USER:-}" ]]; then
  unset SBATCH_EXTRA 2>/dev/null || true
  SBATCH_EXTRA="--partition=${SBATCH_PARTITION:-dgx} --nodes=1 --nodelist=${SBATCH_NODES:-hpc-dgx-b-2,hpc-dgx-b-3} --gres=gpu:1"
fi
export SBATCH_EXTRA
# shellcheck source=/dev/null
source "${_BOOT}/../../echohate_data_paths.sh"
echohate_align_data_root
export RANDOM_STATE
# CL ablation: folder wins for seed_*_wCL / seed_*_woCL (same contract as cue submit).
# DISABLE_CE_CONTRASTIVE=1 still forces CE-only. Export "none" not empty (Slurm-safe).
if [[ "${DISABLE_CE_CONTRASTIVE:-0}" == "1" ]]; then
  CE_CONTRASTIVE_LAMBDA=""
elif [[ "${CE_CONTRASTIVE_LAMBDA_EXPLICIT:-0}" == "1" ]]; then
  : # keep caller CE_CONTRASTIVE_LAMBDA
else
  case "${ECHOHATE_DATA_SEED_DIR:-}" in
    *_woCL) CE_CONTRASTIVE_LAMBDA="" ;;
    *_wCL) CE_CONTRASTIVE_LAMBDA=0.5 ;;
    *) CE_CONTRASTIVE_LAMBDA="${CE_CONTRASTIVE_LAMBDA:-0.5}" ;;
  esac
fi
if [[ -z "${CE_CONTRASTIVE_LAMBDA}" ]]; then
  CE_CONTRASTIVE_LAMBDA=none
fi
case "${ECHOHATE_DATA_SEED_DIR:-}" in
  *_woCL)
    if [[ "${CE_CONTRASTIVE_LAMBDA}" != "none" && "${CE_CONTRASTIVE_LAMBDA}" != "off" ]]; then
      echo "ERROR: ${ECHOHATE_DATA_SEED_DIR} is woCL but CE_CONTRASTIVE_LAMBDA='${CE_CONTRASTIVE_LAMBDA}'" >&2
      exit 1
    fi
    ;;
  *_wCL)
    if [[ "${CE_CONTRASTIVE_LAMBDA}" == "none" || "${CE_CONTRASTIVE_LAMBDA}" == "off" || -z "${CE_CONTRASTIVE_LAMBDA}" ]]; then
      echo "ERROR: ${ECHOHATE_DATA_SEED_DIR} is wCL but CE_CONTRASTIVE_LAMBDA is empty/none" >&2
      exit 1
    fi
    ;;
esac
echo "[submit_roberta_baseline] CE_CONTRASTIVE_LAMBDA='${CE_CONTRASTIVE_LAMBDA}' (seed=${ECHOHATE_DATA_SEED_DIR})"
CONTRASTIVE_MARGIN="${CONTRASTIVE_MARGIN:-0.75}"
TRACK_DYNAMICS="${TRACK_DYNAMICS:-1}"
SKIP_TEST_INFER="${SKIP_TEST_INFER:-0}"
TRAIN_ON_TRAIN_VAL="${TRAIN_ON_TRAIN_VAL:-0}"
EPOCHS_FROM_TRAIN_BASELINE="${EPOCHS_FROM_TRAIN_BASELINE:-0}"
# from80 (default) or 0.5 — only used when TRAIN_ON_TRAIN_VAL=1.
TRAINVAL_APPLIED_THRESHOLD="${TRAINVAL_APPLIED_THRESHOLD:-from80}"
NO_EVAL_EACH_EPOCH="${NO_EVAL_EACH_EPOCH:-0}"
# Map default baseline dir → the 4 trainval protocol folders.
if [[ "${TRAIN_ON_TRAIN_VAL}" == "1" ]]; then
  case "${TRAINVAL_APPLIED_THRESHOLD}" in
    0.5|from80) ;;
    *)
      echo "ERROR: TRAINVAL_APPLIED_THRESHOLD must be 'from80' or '0.5' (got: ${TRAINVAL_APPLIED_THRESHOLD})" >&2
      exit 1
      ;;
  esac
  _rb_auto_dirs=(
    roberta_large_baseline
    roberta_large_baseline_trainval
    roberta_large_baseline_trainval_bestepoch
    roberta_large_baseline_trainval_from80tau
    roberta_large_baseline_trainval_bestepoch_from80tau
  )
  _rb_is_auto=0
  for _d in "${_rb_auto_dirs[@]}"; do
    if [[ "${ECHOHATE_ROBERTA_BASELINE_DIR}" == "${_d}" ]]; then
      _rb_is_auto=1
      break
    fi
  done
  if [[ "${_rb_is_auto}" == "1" ]]; then
    if [[ "${TRAINVAL_APPLIED_THRESHOLD}" == "0.5" ]]; then
      if [[ "${EPOCHS_FROM_TRAIN_BASELINE}" == "1" ]]; then
        ECHOHATE_ROBERTA_BASELINE_DIR="roberta_large_baseline_trainval_bestepoch"
      else
        ECHOHATE_ROBERTA_BASELINE_DIR="roberta_large_baseline_trainval"
      fi
    else
      if [[ "${EPOCHS_FROM_TRAIN_BASELINE}" == "1" ]]; then
        ECHOHATE_ROBERTA_BASELINE_DIR="roberta_large_baseline_trainval_bestepoch_from80tau"
      else
        ECHOHATE_ROBERTA_BASELINE_DIR="roberta_large_baseline_trainval_from80tau"
      fi
    fi
  fi
fi
export ECHOHATE_ROBERTA_BASELINE_DIR
export TRAIN_ON_TRAIN_VAL
export EPOCHS_FROM_TRAIN_BASELINE
export TRAINVAL_APPLIED_THRESHOLD
export NO_EVAL_EACH_EPOCH
export SKIP_TEST_INFER
export TRACK_DYNAMICS

_echohate_root() {
  local root="${_BOOT}/../../.."
  if [[ -f "${root}/pipeline/stage2/roberta_baseline.py" ]]; then
    cd "${root}" && pwd
    return 0
  fi
  return 1
}

# Shared training defaults (NUM_EPOCHS, LR, …) — same file Slurm sources on HPC.
ECHOHATE_ROOT="$(_echohate_root)"
export ECHOHATE_ROOT
# shellcheck source=/dev/null
source "${_BOOT}/../../training_hyper_defaults.sh"

_discover_datasets_with_splits() {
  local root data_dir ds
  root="$(_echohate_root)"
  data_dir="${root}/data/${ECHOHATE_DATA_SEED_DIR}"
  if [[ ! -d "${data_dir}" ]]; then
    echo "ERROR: missing ${data_dir}" >&2
    exit 1
  fi
  for ds in "${data_dir}"/*; do
    [[ -d "${ds}" ]] || continue
    ds="$(basename "${ds}")"
    if [[ -f "${data_dir}/${ds}/${ECHOHATE_SPLITTED_DATA_SUBDIR}/train.csv" ]]; then
      printf '%s\n' "${ds}"
    fi
  done
}

_print_baseline_hyper() {
  local dataset="${1:-}"
  local out_dir="${2:-}"
  echo "------------------------------------------"
  if [[ -n "${dataset}" ]]; then
    echo "RoBERTa baseline hyper — ${dataset}"
  else
    echo "RoBERTa baseline hyper (all jobs)"
  fi
  echo "  paths:"
  echo "    ECHOHATE_DATA_SEED_DIR=${ECHOHATE_DATA_SEED_DIR}"
  echo "    DATA_ROOT=${DATA_ROOT}"
  echo "    ECHOHATE_SPLITTED_DATA_SUBDIR=${ECHOHATE_SPLITTED_DATA_SUBDIR}"
  [[ -n "${out_dir}" ]] && echo "    OUTPUT_DIR=${out_dir}"
  echo "  model / protocol:"
  echo "    MODEL_NAME=${MODEL_NAME}  full_finetune=1  (no LoRA)"
  if [[ "${TRAIN_ON_TRAIN_VAL}" == "1" ]]; then
    echo "    train=train+validation  tune=none (fixed epochs; thr=${TRAINVAL_APPLIED_THRESHOLD})  test=infer unless SKIP_TEST_INFER=1"
  else
    echo "    train=train.csv  tune=validation.csv  test=infer unless SKIP_TEST_INFER=1"
  fi
  echo "    TRAIN_ON_TRAIN_VAL=${TRAIN_ON_TRAIN_VAL}"
  echo "    TRAINVAL_APPLIED_THRESHOLD=${TRAINVAL_APPLIED_THRESHOLD}"
  echo "    EPOCHS_FROM_TRAIN_BASELINE=${EPOCHS_FROM_TRAIN_BASELINE}"
  echo "    eval_each_epoch=$([[ "${NO_EVAL_EACH_EPOCH:-0}" == "1" || "${TRAIN_ON_TRAIN_VAL}" == "1" ]] && echo 0 || echo 1)"
  echo "    tune_threshold_on=$([[ "${TRAIN_ON_TRAIN_VAL}" == "1" ]] && echo "trainval:${TRAINVAL_APPLIED_THRESHOLD}" || echo validation)  infer_test=$([[ "${SKIP_TEST_INFER}" == "1" ]] && echo 0 || echo 1)"
  echo "    TRACK_DYNAMICS=${TRACK_DYNAMICS}"
  echo "    ECHOHATE_ROBERTA_BASELINE_DIR=${ECHOHATE_ROBERTA_BASELINE_DIR}"
  echo "  training:"
  if [[ "${EPOCHS_FROM_TRAIN_BASELINE}" == "1" ]]; then
    echo "    NUM_EPOCHS=<from train-only best_epoch at job start>  BATCH_SIZE=${BATCH_SIZE}  EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  else
    echo "    NUM_EPOCHS=${NUM_EPOCHS}  BATCH_SIZE=${BATCH_SIZE}  EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  fi
  echo "    LEARNING_RATE=${LEARNING_RATE}  WEIGHT_DECAY=${WEIGHT_DECAY}  WARMUP_RATIO=${WARMUP_RATIO}"
  echo "    MAX_LENGTH=${MAX_LENGTH}  RANDOM_STATE=${RANDOM_STATE}"
  echo "    POS_WEIGHT_MODE=${POS_WEIGHT_MODE}  CLASS_WEIGHT_TARGET=${CLASS_WEIGHT_TARGET}"
  echo "  loss:"
  if [[ -n "${CE_CONTRASTIVE_LAMBDA}" ]]; then
    echo "    CE + contrastive  CE_CONTRASTIVE_LAMBDA=${CE_CONTRASTIVE_LAMBDA}  CONTRASTIVE_MARGIN=${CONTRASTIVE_MARGIN}"
  else
    echo "    CE only (no contrastive aux loss)"
  fi
  echo "  hpc:"
  echo "    CONDA_ENV=${CONDA_ENV}  HPC=${HPC_USER}@${HPC_HOST}"
  echo "    SBATCH_EXTRA=${SBATCH_EXTRA}"
  echo "------------------------------------------"
}

_baseline_sbatch_export() {
  # No quotes around values — sbatch --export treats 'name' as part of the value.
  printf '%s' "ALL,DATASET=${1},DATA_ROOT=${DATA_ROOT},OUTPUT_DIR=${2},MODEL_NAME=${MODEL_NAME},CONDA_ENV=${CONDA_ENV},CREATE_ENV=${CREATE_ENV:-0},RANDOM_STATE=${RANDOM_STATE},ECHOHATE_DATA_SEED_DIR=${ECHOHATE_DATA_SEED_DIR},ECHOHATE_SPLITTED_DATA_SUBDIR=${ECHOHATE_SPLITTED_DATA_SUBDIR},SKIP_TEST_INFER=${SKIP_TEST_INFER},TRAIN_ON_TRAIN_VAL=${TRAIN_ON_TRAIN_VAL},EPOCHS_FROM_TRAIN_BASELINE=${EPOCHS_FROM_TRAIN_BASELINE},TRAINVAL_APPLIED_THRESHOLD=${TRAINVAL_APPLIED_THRESHOLD},NUM_EPOCHS=${NUM_EPOCHS},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},LEARNING_RATE=${LEARNING_RATE},WEIGHT_DECAY=${WEIGHT_DECAY},WARMUP_RATIO=${WARMUP_RATIO},MAX_LENGTH=${MAX_LENGTH},POS_WEIGHT_MODE=${POS_WEIGHT_MODE},CLASS_WEIGHT_TARGET=${CLASS_WEIGHT_TARGET},NO_EVAL_EACH_EPOCH=${NO_EVAL_EACH_EPOCH},CE_CONTRASTIVE_LAMBDA=${CE_CONTRASTIVE_LAMBDA},CONTRASTIVE_MARGIN=${CONTRASTIVE_MARGIN},TRACK_DYNAMICS=${TRACK_DYNAMICS}"
}

_submit_one() {
  local dataset="$1"
  local out_dir="${OUTPUT_DIR:-data/${ECHOHATE_DATA_SEED_DIR}/${dataset}/${ECHOHATE_ROBERTA_BASELINE_DIR}}"
  echohate_assert_seed_paths_consistent "${out_dir}" || exit 1
  local export_vars
  export_vars="$(_baseline_sbatch_export "${dataset}" "${out_dir}")"
  echo "=========================================="
  echo "Submit baseline: ${dataset}"
  _print_baseline_hyper "${dataset}" "${out_dir}"
  echo "=========================================="
  hpc_mkdir_remote_from_local "${HPC_DIR}/${out_dir}"
  # Clear stale login-shell CUDA_VISIBLE_DEVICES so sbatch --export=ALL does not break DGX jobs
  # (same as submit_llm_from_local.sh / submit_abstract_from_local.sh).
  if ! hpc_ssh_sbatch_export_all "${HPC_DIR}" "${SLURM_SCRIPT}" "${export_vars}" \
    "--job-name=${ROBERTA_SBATCH_JOB_PREFIX:-rb_}${dataset} ${SBATCH_EXTRA}" "unset CUDA_VISIBLE_DEVICES"; then
    echo "ERROR: sbatch failed for ${dataset}" >&2
    exit 1
  fi
}

if [[ "${ALL_DATASETS:-0}" == "1" ]]; then
  _DATASET_LIST=()
  while IFS= read -r _ds_line; do
    [[ -n "${_ds_line}" ]] || continue
    _DATASET_LIST+=("${_ds_line}")
  done < <(_discover_datasets_with_splits | sort)
elif [[ -n "${DATASETS:-}" ]]; then
  IFS=',' read -r -a _DATASET_LIST <<< "${DATASETS}"
else
  _DATASET_LIST=("${DATASET:-IHC}")
fi

if [[ ${#_DATASET_LIST[@]} -eq 0 ]]; then
  echo "ERROR: no datasets to submit (check DATASETS / ALL_DATASETS / splits under data/${ECHOHATE_DATA_SEED_DIR}/)" >&2
  exit 1
fi

echo ""
_print_baseline_hyper "" ""
echo "Datasets (${#_DATASET_LIST[@]}): ${_DATASET_LIST[*]}"
echo ""

if [[ "${SKIP_DATA_SYNC:-0}" == "1" ]]; then
  echo "SKIP_DATA_SYNC=1 — not rsyncing data/${ECHOHATE_DATA_SEED_DIR}"
else
  echo "Sync code + data/${ECHOHATE_DATA_SEED_DIR} to HPC (once) ..."
  if [[ "${ALL_DATASETS:-0}" == "1" || -n "${DATASETS:-}" ]]; then
    SYNC_ALL_DATA=1 ECHOHATE_DATA_SEED_DIR="${ECHOHATE_DATA_SEED_DIR}" \
      bash "${_BOOT}/../../sync_echohate_code_and_data_to_hpc_from_local.sh" || {
      echo "ERROR: code/data sync to HPC failed (job would run stale job_submission/hpc_env_deakin.sh)." >&2
      exit 1
    }
  else
    DATASET="${_DATASET_LIST[0]}" \
      bash "${_BOOT}/../../sync_echohate_code_and_data_to_hpc_from_local.sh" "${_DATASET_LIST[0]}" "${ECHOHATE_DATA_SEED_DIR}" || {
      echo "ERROR: code/data sync to HPC failed (job would run stale job_submission/hpc_env_deakin.sh)." >&2
      exit 1
    }
  fi
fi

for ds in "${_DATASET_LIST[@]}"; do
  ds="${ds// /}"
  [[ -n "${ds}" ]] || continue
  _submit_one "${ds}"
done

echo "Done. Submitted ${#_DATASET_LIST[@]} job(s). Check: ssh ${HPC_USER}@${HPC_HOST} squeue -u ${HPC_USER}"
