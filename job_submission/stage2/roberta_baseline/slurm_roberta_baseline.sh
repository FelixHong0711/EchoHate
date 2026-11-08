#!/bin/bash
#SBATCH --job-name=roberta_baseline
#SBATCH --output=roberta_baseline_%j.out
#SBATCH --error=roberta_baseline_%j.err
# Partition from sbatch CLI (submit script): dgx on hpclogin_dgx, gpu on hpc2.
# Do not set #SBATCH --partition here — invalid partition names break submission.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Stage 2: standalone RoBERTa FFT baseline (no LLM cues).

set -euo pipefail
# Always run from the EchoHate checkout under the *echohate* path name.
# On HPC, experiment/echohate may historically be a symlink to co2hate (same inode). Prefer the
# echohate path string so cwd / DATA_ROOT / metrics never say /co2hate/. Never keep a stale
# ECHOHATE_ROOT from --export=ALL (${ECHOHATE_ROOT:-...} would preserve a bad value).
_EXP_PARENT="/home/remote/hohuu/RA/experiment"
_ECHOHATE_ROOT_DEFAULT="${_EXP_PARENT}/echohate"
_rewrite_co2hate_to_echohate() {
  # Rewrite path *strings* only when .../co2hate is the same checkout as .../echohate.
  local p="$1"
  local phys_c phys_e
  phys_c="$(readlink -f "${_EXP_PARENT}/co2hate" 2>/dev/null || true)"
  phys_e="$(readlink -f "${_EXP_PARENT}/echohate" 2>/dev/null || true)"
  if [[ -n "${phys_c}" && -n "${phys_e}" && "${phys_c}" == "${phys_e}" ]]; then
    printf '%s' "${p//\/co2hate\//\/echohate\/}"
  else
    printf '%s' "${p}"
  fi
}
if [[ -n "${ECHOHATE_ROOT:-}" && "${ECHOHATE_ROOT}" == *"/co2hate"* ]]; then
  echo "WARN: rewriting stale ECHOHATE_ROOT=${ECHOHATE_ROOT} → echohate path." >&2
  ECHOHATE_ROOT="$(_rewrite_co2hate_to_echohate "${ECHOHATE_ROOT}")"
fi
# Prefer the checkout we sbatch'd from (usually /scratch/.../echohate). Do NOT
# overwrite a valid SLURM_SUBMIT_DIR with home: on hpclogin, home/echohate often
# symlinks to a stale co2hate tree missing job_submission/hpc_env_deakin.sh.
_SCRATCH_ECHOHATE="/scratch/hohuu/RA/experiment/echohate"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pipeline/stage2/roberta_baseline.py" ]]; then
  ECHOHATE_ROOT="$(_rewrite_co2hate_to_echohate "${SLURM_SUBMIT_DIR}")"
elif [[ -f "${_SCRATCH_ECHOHATE}/pipeline/stage2/roberta_baseline.py" ]]; then
  ECHOHATE_ROOT="${_SCRATCH_ECHOHATE}"
else
  ECHOHATE_ROOT="${_ECHOHATE_ROOT_DEFAULT}"
fi
# If the chosen root is broken (home→co2hate without hpc_env), fall back to scratch.
if [[ ! -f "${ECHOHATE_ROOT}/job_submission/hpc_env_deakin.sh" ]] \
  && [[ -f "${_SCRATCH_ECHOHATE}/job_submission/hpc_env_deakin.sh" ]]; then
  echo "WARN: ${ECHOHATE_ROOT} missing job_submission/hpc_env_deakin.sh; using ${_SCRATCH_ECHOHATE}" >&2
  ECHOHATE_ROOT="${_SCRATCH_ECHOHATE}"
fi
cd "${ECHOHATE_ROOT}"
export ECHOHATE_ROOT
# Prefer this checkout's pipeline/ over any leftover PYTHONPATH (e.g. editable install).
# Re-applied again AFTER conda activate — activate often clobbers / prepends PYTHONPATH.
_echohate_pin_pythonpath() {
  unset PYTHONPATH
  export PYTHONPATH="${ECHOHATE_ROOT}"
  export PYTHONNOUSERSITE=1
}
_echohate_pin_pythonpath
_PWD_L="$(pwd -L 2>/dev/null || pwd)"
_PWD_P="$(pwd -P 2>/dev/null || pwd)"
echo "cwd=${_PWD_L}  cwd_phys=${_PWD_P}  PYTHONPATH=${PYTHONPATH}  ECHOHATE_ROOT=${ECHOHATE_ROOT}"
# Refuse only if physical cwd is some *other* project tree (not the echohate/co2hate alias).
_PHYS_OK="$(readlink -f "${ECHOHATE_ROOT}" 2>/dev/null || true)"
if [[ -n "${_PHYS_OK}" && "${_PWD_P}" != "${_PHYS_OK}"* ]]; then
  echo "ERROR: physical cwd (${_PWD_P}) is not under ECHOHATE_ROOT phys (${_PHYS_OK})." >&2
  exit 1
fi
if [[ ! -f "${ECHOHATE_ROOT}/pipeline/stage2/roberta_baseline.py" ]]; then
  echo "ERROR: missing ${ECHOHATE_ROOT}/pipeline/stage2/roberta_baseline.py" >&2
  exit 1
fi
# Relative DATA_ROOT/OUTPUT_DIR must resolve under echohate (submit often exports DATA_ROOT=data/seed_*)
if [[ -n "${DATA_ROOT:-}" && "${DATA_ROOT}" != /* ]]; then
  export DATA_ROOT="${ECHOHATE_ROOT}/${DATA_ROOT}"
fi
if [[ -n "${OUTPUT_DIR:-}" && "${OUTPUT_DIR}" != /* ]]; then
  export OUTPUT_DIR="${ECHOHATE_ROOT}/${OUTPUT_DIR}"
fi
export DATA_ROOT="$(_rewrite_co2hate_to_echohate "${DATA_ROOT:-}")"
export OUTPUT_DIR="$(_rewrite_co2hate_to_echohate "${OUTPUT_DIR:-}")"
# If still under /co2hate/ and that is NOT the same inode as echohate, abort.
_phys_c="$(readlink -f "${_EXP_PARENT}/co2hate" 2>/dev/null || true)"
_phys_e="$(readlink -f "${_EXP_PARENT}/echohate" 2>/dev/null || true)"
if { [[ "${DATA_ROOT:-}" == *"/co2hate/"* ]] || [[ "${OUTPUT_DIR:-}" == *"/co2hate/"* ]]; } \
  && [[ -n "${_phys_c}" && -n "${_phys_e}" && "${_phys_c}" != "${_phys_e}" ]]; then
  echo "ERROR: DATA_ROOT/OUTPUT_DIR points at a separate co2hate tree (DATA_ROOT=${DATA_ROOT:-} OUTPUT_DIR=${OUTPUT_DIR:-})." >&2
  exit 1
fi

export HPC_REQUIRE_CUDA="${HPC_REQUIRE_CUDA:-1}"
export HPC_MODULE_PURGE="${HPC_MODULE_PURGE:-0}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

set +u
source "${ECHOHATE_ROOT}/job_submission/hpc_env_deakin.sh"
set -u

# conda activate may reset PYTHONPATH; editable co2hate .pth/.egg-link can still win — scrub it.
_echohate_pin_pythonpath
_RUN_MOD="${ECHOHATE_ROOT}/job_submission/stage2/roberta_baseline/echohate_run_module.py"
if [[ ! -f "${_RUN_MOD}" ]]; then
  echo "ERROR: missing ${_RUN_MOD} (sync job_submission to HPC)." >&2
  exit 1
fi
_rb_file="$(
  python -c "
import os, sys
from pathlib import Path
root = os.environ['ECHOHATE_ROOT']
root_phys = str(Path(root).resolve())
cleaned = []
for p in sys.path:
    if not p:
        continue
    norm = p.replace(chr(92), '/')
    # Drop other checkouts named co2hate only when they are a *different* tree.
    if '/co2hate' in norm:
        try:
            if str(Path(p).resolve()) != root_phys and not str(Path(p).resolve()).startswith(root_phys + os.sep):
                continue
        except Exception:
            continue
    cleaned.append(p)
sys.path[:] = cleaned
sys.path.insert(0, root)
import pipeline.stage2.roberta_baseline as m
print(m.__file__)
"
)"
echo "roberta_baseline=${_rb_file}"
_rb_phys="$(readlink -f "${_rb_file}" 2>/dev/null || true)"
_root_phys="$(readlink -f "${ECHOHATE_ROOT}" 2>/dev/null || true)"
if [[ -z "${_rb_phys}" || -z "${_root_phys}" || "${_rb_phys}" != "${_root_phys}"* ]]; then
  echo "ERROR: imported roberta_baseline outside ECHOHATE_ROOT (${_rb_file} → ${_rb_phys})." >&2
  echo "  Uninstall editable co2hate/echohate installs or scrub site-packages egg-link/.pth." >&2
  exit 1
fi
# Prefer reporting the echohate path string when aliasing the same inode.
_rb_show="$(_rewrite_co2hate_to_echohate "${_rb_file}")"
echo "roberta_baseline_canon=${_rb_show}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L 2>/dev/null || nvidia-smi 2>/dev/null || true
fi
if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: roberta_baseline requires CUDA (torch.cuda.is_available() is False). Check module load / --gres=gpu:1." >&2
  exit 1
fi
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))" 2>/dev/null || true

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# shellcheck source=/dev/null
source "${ECHOHATE_ROOT}/job_submission/echohate_project_env.sh"
# shellcheck source=/dev/null
source "${ECHOHATE_ROOT}/job_submission/training_hyper_defaults.sh"
# shellcheck source=/dev/null
source "${ECHOHATE_ROOT}/job_submission/echohate_data_paths.sh"

DATASET="${DATASET:?missing DATASET}"
echohate_align_data_root
MODEL_NAME="${MODEL_NAME:-roberta-large}"
OUTPUT_DIR="${OUTPUT_DIR:?missing OUTPUT_DIR}"
echohate_assert_seed_paths_consistent "${OUTPUT_DIR}" || exit 1
# Folder path wins for wCL/woCL ablation (overrides stale --export=ALL).
# Sentinel "none"/"off" from submit → CE-only. Default λ=0.5 only when unset AND not woCL.
_p_cl="${OUTPUT_DIR:-}:${DATA_ROOT:-}:${ECHOHATE_DATA_SEED_DIR:-}"
case "${_p_cl}" in
  *_woCL*)
    CE_CONTRASTIVE_LAMBDA=""
    ;;
  *_wCL*)
    if [[ -z "${CE_CONTRASTIVE_LAMBDA:-}" || "${CE_CONTRASTIVE_LAMBDA}" == "none" || "${CE_CONTRASTIVE_LAMBDA}" == "off" ]]; then
      CE_CONTRASTIVE_LAMBDA=0.5
    fi
    ;;
  *)
    if [[ "${CE_CONTRASTIVE_LAMBDA:-}" == "none" || "${CE_CONTRASTIVE_LAMBDA:-}" == "off" ]]; then
      CE_CONTRASTIVE_LAMBDA=""
    elif [[ -z "${CE_CONTRASTIVE_LAMBDA+x}" ]]; then
      # Default λ=0.5 only when unset. Explicit empty must stay empty for CE-only.
      CE_CONTRASTIVE_LAMBDA=0.5
    fi
    ;;
esac
if [[ "${CE_CONTRASTIVE_LAMBDA:-}" == "none" || "${CE_CONTRASTIVE_LAMBDA:-}" == "off" ]]; then
  CE_CONTRASTIVE_LAMBDA=""
fi
if [[ "${OUTPUT_DIR:-}" == *"_woCL"* && -n "${CE_CONTRASTIVE_LAMBDA:-}" ]]; then
  echo "ERROR: OUTPUT_DIR is woCL but CE_CONTRASTIVE_LAMBDA='${CE_CONTRASTIVE_LAMBDA}'" >&2
  exit 1
fi
if [[ "${OUTPUT_DIR:-}" == *"_wCL"* && "${OUTPUT_DIR:-}" != *"_woCL"* && -z "${CE_CONTRASTIVE_LAMBDA:-}" ]]; then
  echo "ERROR: OUTPUT_DIR is wCL but CE_CONTRASTIVE_LAMBDA is empty" >&2
  exit 1
fi
CONTRASTIVE_MARGIN="${CONTRASTIVE_MARGIN:-0.75}"
TRACK_DYNAMICS="${TRACK_DYNAMICS:-1}"
SKIP_TEST_INFER="${SKIP_TEST_INFER:-0}"
TRAIN_ON_TRAIN_VAL="${TRAIN_ON_TRAIN_VAL:-0}"
EPOCHS_FROM_TRAIN_BASELINE="${EPOCHS_FROM_TRAIN_BASELINE:-0}"
TRAINVAL_APPLIED_THRESHOLD="${TRAINVAL_APPLIED_THRESHOLD:-from80}"
NO_EVAL_EACH_EPOCH="${NO_EVAL_EACH_EPOCH:-0}"

echo "========== roberta_baseline run config =========="
echo "DATASET=${DATASET}  DATA_ROOT=${DATA_ROOT}  OUTPUT_DIR=${OUTPUT_DIR}"
echo "ECHOHATE_SPLITTED_DATA_SUBDIR=${ECHOHATE_SPLITTED_DATA_SUBDIR:-splitted_data_80_10_10}"
echo "MODEL_NAME=${MODEL_NAME}  RANDOM_STATE=${RANDOM_STATE}"
echo "NUM_EPOCHS=${NUM_EPOCHS}  BATCH_SIZE=${BATCH_SIZE}  EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "LEARNING_RATE=${LEARNING_RATE}  WEIGHT_DECAY=${WEIGHT_DECAY}  WARMUP_RATIO=${WARMUP_RATIO}  MAX_LENGTH=${MAX_LENGTH}"
echo "POS_WEIGHT_MODE=${POS_WEIGHT_MODE}  CLASS_WEIGHT_TARGET=${CLASS_WEIGHT_TARGET}"
echo "CE_CONTRASTIVE_LAMBDA=${CE_CONTRASTIVE_LAMBDA}  CONTRASTIVE_MARGIN=${CONTRASTIVE_MARGIN}"
echo "TRACK_DYNAMICS=${TRACK_DYNAMICS}  SKIP_TEST_INFER=${SKIP_TEST_INFER}"
echo "TRAIN_ON_TRAIN_VAL=${TRAIN_ON_TRAIN_VAL}  EPOCHS_FROM_TRAIN_BASELINE=${EPOCHS_FROM_TRAIN_BASELINE}  TRAINVAL_APPLIED_THRESHOLD=${TRAINVAL_APPLIED_THRESHOLD}  NO_EVAL_EACH_EPOCH=${NO_EVAL_EACH_EPOCH}"
echo "================================================="

mkdir -p "${OUTPUT_DIR}"
_hyper_file="${OUTPUT_DIR}/run_hyperparams.txt"
{
  echo "# roberta_baseline — written at job start ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "DATASET=${DATASET}"
  echo "ECHOHATE_DATA_SEED_DIR=${ECHOHATE_DATA_SEED_DIR:-}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "ECHOHATE_SPLITTED_DATA_SUBDIR=${ECHOHATE_SPLITTED_DATA_SUBDIR:-splitted_data_80_10_10}"
  echo "MODEL_NAME=${MODEL_NAME}"
  echo "RANDOM_STATE=${RANDOM_STATE}"
  echo "NUM_EPOCHS=${NUM_EPOCHS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "LEARNING_RATE=${LEARNING_RATE}"
  echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
  echo "WARMUP_RATIO=${WARMUP_RATIO}"
  echo "MAX_LENGTH=${MAX_LENGTH}"
  echo "POS_WEIGHT_MODE=${POS_WEIGHT_MODE}"
  echo "CLASS_WEIGHT_TARGET=${CLASS_WEIGHT_TARGET}"
  echo "NO_EVAL_EACH_EPOCH=${NO_EVAL_EACH_EPOCH}"
  echo "TRAIN_ON_TRAIN_VAL=${TRAIN_ON_TRAIN_VAL}"
  echo "EPOCHS_FROM_TRAIN_BASELINE=${EPOCHS_FROM_TRAIN_BASELINE}"
  echo "TRAINVAL_APPLIED_THRESHOLD=${TRAINVAL_APPLIED_THRESHOLD}"
  echo "CE_CONTRASTIVE_LAMBDA=${CE_CONTRASTIVE_LAMBDA}"
  echo "CONTRASTIVE_MARGIN=${CONTRASTIVE_MARGIN}"
  echo "TRACK_DYNAMICS=${TRACK_DYNAMICS}"
  echo "SKIP_TEST_INFER=${SKIP_TEST_INFER}"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
} > "${_hyper_file}"
echo "Wrote ${_hyper_file}"

CMD=(python "${_RUN_MOD}" pipeline.stage2.roberta_baseline
  --require_gpu
  --dataset "${DATASET}"
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --model_name "${MODEL_NAME}"
  --num_epochs "${NUM_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --max_length "${MAX_LENGTH}"
  --random_state "${RANDOM_STATE}"
  --pos_weight_mode "${POS_WEIGHT_MODE}"
  --class_weight_target "${CLASS_WEIGHT_TARGET}"
)

if [[ "${SKIP_TEST_INFER:-0}" == "1" ]]; then
  CMD+=(--skip-test-infer)
fi

if [[ "${TRAIN_ON_TRAIN_VAL:-0}" == "1" ]]; then
  CMD+=(--train-on-train-val)
  CMD+=(--trainval-applied-threshold "${TRAINVAL_APPLIED_THRESHOLD}")
fi

if [[ "${EPOCHS_FROM_TRAIN_BASELINE:-0}" == "1" ]]; then
  CMD+=(--epochs-from-train-baseline)
  if [[ -n "${EPOCHS_FROM_METRIC_SUMMARY:-}" ]]; then
    CMD+=(--epochs-from-metric-summary "${EPOCHS_FROM_METRIC_SUMMARY}")
  fi
fi

if [[ "${NO_EVAL_EACH_EPOCH:-0}" == "1" ]]; then
  CMD+=(--no-eval-each-epoch)
fi

if [[ "${TRACK_DYNAMICS:-1}" == "1" ]]; then
  CMD+=(--track-dynamics)
fi

if [[ -n "${CE_CONTRASTIVE_LAMBDA}" ]]; then
  CMD+=(--ce-contrastive-lambda "${CE_CONTRASTIVE_LAMBDA}")
  CMD+=(--contrastive-margin "${CONTRASTIVE_MARGIN}")
fi

"${CMD[@]}"
echo "Done roberta_baseline → ${OUTPUT_DIR}"
