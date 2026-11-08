#!/usr/bin/env bash
# Adapt toy_prompt hard LLM preds → mech-distill teacher CSVs, sync to HPC, submit mech-only FFT.
#
# Prereq locally (per encoder run dir under data/<seed>/<DS>/):
#   selection_gmm/hard_train_for_toy_prompt.csv  (or selected_hard_rows.csv)
#   toy_prompt_<mode>_hard_<phi|qwen><tag>_batch<N>/llm_test_predictions.csv
#
# Writes under the same run dir:
#   llm_<fam>_selected_hard_nocontext<TEACHER_OUTPUT_TAG>/selected_k_llm_predictions.csv
#
# Examples:
#   # HatEval eacl — all 4 encoders × phi+qwen mech-only (≈48 jobs)
#   export HPC_RA=/home/remote/hohuu/RA HPC_RA_RESOLVE=0
#   DATASETS=HatEval ECHOHATE_DATA_SEED_DIR=eacl_seed0 HARD_ENCODERS=all \
#     bash job_submission/stage2/cue_mech_attnmask/submit_mech_only_from_toy_prompt_hard_hpclogin_dgx_from_local.sh
#
#   # One encoder, adapt only (no sbatch)
#   ADAPT_ONLY=1 DATASETS=HatEval HARD_ENCODERS=roberta_large_baseline_wCL \
#     bash job_submission/stage2/cue_mech_attnmask/submit_mech_only_from_toy_prompt_hard_from_local.sh
#
# Env:
#   DATASETS              comma list (default HatEval)
#   ECHOHATE_DATA_SEED_DIR  default eacl_seed0
#   HARD_ENCODERS         comma list or "all" (auto-discover */selection_gmm)
#   TEACHER_FAMILIES      phi,qwen (default)
#   TEACHER_OUTPUT_TAG    default _eacl_batch8 → llm_*_selected_hard_nocontext_eacl_batch8
#   TOY_OUTPUT_TAG        default eacl → toy_prompt_*_hard_*_eacl_batch8
#   TOY_HARD_TOKEN        hard (default) or hard_prompt_GT (GT-conditioned toy dirs)
#   PROMPT_MODE           nocontext (default)
#   LLM_BATCH_SIZE        8 (must match toy_prompt infer dirs)
#   TEACHER_MODES         nocontext (default; mech distill teacher mode)
#   ENCODER_RUN_TAG       _mech_only (default)
#   GRID_PRESET           full|tiny (default full → 6 λ_mech / fam / mode)
#   CE_CONTRASTIVE_LAMBDA  unset → infer from encoder dirname (*_wCL→0.5, *_woCL→none)
#   ADAPT_ONLY=1          adapt + optional sync; skip sbatch
#   SKIP_ADAPT=1          skip adapt (teachers already on disk)
#   SYNC_TO_HPC=1         rsync teachers + splits + selection_gmm (default 1)
#   HPC_SKIP_CODE_RSYNC=1 skip full code rsync in mech submitter
#
set -euo pipefail

_BOOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_BOOT}/../../hpc_defaults_from_local.sh"
unset HPC_SSH_CONN_OPTS 2>/dev/null || true
hpc_apply_ra_default_or_resolve
# shellcheck source=/dev/null
source "${_BOOT}/../../echohate_hpc_password_env.sh"
# shellcheck source=/dev/null
source "${_BOOT}/../../echohate_hf_token_env.sh"

# Preserve intentional caller grids/tags before ambient wipe.
_SAVED_CUE_LAMBDA_MECH_GRID="${CUE_LAMBDA_MECH_GRID:-}"
_SAVED_E_INIT_GRID="${E_INIT_GRID:-}"
_SAVED_A_INIT_GRID="${A_INIT_GRID:-}"
_SAVED_T_INIT_GRID="${T_INIT_GRID:-}"
_SAVED_E_SCALE_MODE="${E_SCALE_MODE:-}"
_SAVED_LAMBDA_MECH_GRID="${LAMBDA_MECH_GRID:-}"
_SAVED_LAMBDA_TARGET_GRID="${LAMBDA_TARGET_GRID:-}"
_SAVED_LAMBDA_TARGET="${LAMBDA_TARGET:-}"
_SAVED_TARGET_NONE_POLICY="${TARGET_NONE_POLICY:-}"
_SAVED_PROMPT_MODE="${PROMPT_MODE:-}"
_SAVED_ENCODER_RUN_TAG="${ENCODER_RUN_TAG:-}"
_SAVED_APPROACH="${APPROACH:-}"
_SAVED_AMP_POOL="${AMP_POOL:-}"
_SAVED_FUSION_LEVEL="${FUSION_LEVEL:-}"
_SAVED_TEACHER_OUTPUT_TAG="${TEACHER_OUTPUT_TAG:-}"
_SAVED_TOY_OUTPUT_TAG="${TOY_OUTPUT_TAG:-}"
_SAVED_TOY_HARD_TOKEN="${TOY_HARD_TOKEN:-}"
_SAVED_ECHOHATE_DATA_SEED_DIR="${ECHOHATE_DATA_SEED_DIR:-}"
_SAVED_DATASETS="${DATASETS:-}"
_SAVED_HARD_ENCODERS="${HARD_ENCODERS:-}"
_SAVED_SKIP_ADAPT="${SKIP_ADAPT:-}"
_SAVED_TRACK_DYNAMICS="${TRACK_DYNAMICS:-}"
_SAVED_RANDOM_STATE="${RANDOM_STATE:-}"
_SAVED_SEED="${SEED:-}"
_SAVED_LLM_TEACHER_DIR="${LLM_TEACHER_DIR:-}"
_SAVED_AMPVALTEST="${AMPVALTEST:-}"
_SAVED_LLM_CSV_VALIDATION="${LLM_CSV_VALIDATION:-}"
_SAVED_LLM_CSV_TEST="${LLM_CSV_TEST:-}"
_SAVED_RESELECT_BEST_EPOCH="${RESELECT_BEST_EPOCH:-}"
_SAVED_AMPVALTEST_OUTPUT_PREFIX="${AMPVALTEST_OUTPUT_PREFIX:-}"
_SAVED_SPAN_OP="${SPAN_OP:-}"
_SAVED_SPAN_CHANNELS="${SPAN_CHANNELS:-}"
_SAVED_LLM_NORMAL_COL="${LLM_NORMAL_COL:-}"
_SAVED_SPAN_COVERAGE_MAX="${SPAN_COVERAGE_MAX:-}"
_SAVED_ATTN_DEAMP_LAYERS="${ATTN_DEAMP_LAYERS:-}"
_SAVED_HATE_MECH_TAXONOMY="${HATE_MECH_TAXONOMY:-}"
_SAVED_MECH_AUX_MODE="${MECH_AUX_MODE:-}"
_SAVED_PACK_LAMBDA_GRID="${PACK_LAMBDA_GRID:-}"
_SAVED_PACK_SBATCH_TIME="${PACK_SBATCH_TIME:-}"

# Drop stale exports from prior cue/llm waves unless KEEP_AMBIENT_TEACHER_ENV=1.
# Otherwise TEACHER_OUTPUT_TAG=_new_prompt_Qwen_… / LAMBDA_MECH_GRID / seed=42 leak in.
if [[ "${KEEP_AMBIENT_TEACHER_ENV:-0}" != "1" ]]; then
  unset TEACHER_OUTPUT_TAG TEACHER_OUTPUT_TAG_OVERRIDES \
    TEACHER_OUTPUT_TAG_PHI TEACHER_OUTPUT_TAG_QWEN TEACHER_OUTPUT_TAG_LLAMA \
    TEACHER_OUTPUT_TAG_OVERRIDES_PHI TEACHER_OUTPUT_TAG_OVERRIDES_QWEN \
    TEACHER_OUTPUT_TAG_OVERRIDES_LLAMA TEACHER_LM_FAMILY \
    LAMBDA_MECH_GRID CUE_LAMBDA_MECH_GRID CUE_LAMBDA_MECH E_INIT_GRID \
    LAMBDA_TARGET_GRID LAMBDA_TARGET TARGET_NONE_POLICY PROMPT_MODE \
    A_INIT_GRID T_INIT_GRID E_SCALE_MODE \
    STUDENT_MODEL_NAME VALIDATION_ROOT_ENCODER_BN MODEL_NAME \
    CE_CONTRASTIVE_LAMBDA CE_CONTRASTIVE_LAMBDA_EXPLICIT \
    BERT_PHI_NEW_PROMPT_ONLY_DATASETS BERT_PHI_NEW_PROMPT_TAG \
    ENCODER_DATA_ROOT RANDOM_STATE SEED \
    TOY_HARD_TOKEN LLM_SEL_HARD_MID LLM_SEL_HARD_LAYOUT \
    INIT_BACKBONE_FROM INIT_BACKBONE_AUTO INIT_LOAD_CLASSIFIER INIT_LOAD_E \
    LLM_TEACHER_DIR AMPVALTEST LLM_CSV_VALIDATION LLM_CSV_TEST \
    RESELECT_BEST_EPOCH AMPVALTEST_OUTPUT_PREFIX \
    SPAN_OP SPAN_CHANNELS LLM_NORMAL_COL SPAN_COVERAGE_MAX ATTN_DEAMP_LAYERS \
    HATE_MECH_TAXONOMY MECH_AUX_MODE PACK_LAMBDA_GRID PACK_SBATCH_TIME 2>/dev/null || true
fi

[[ -n "${_SAVED_CUE_LAMBDA_MECH_GRID}" ]] && export CUE_LAMBDA_MECH_GRID="${_SAVED_CUE_LAMBDA_MECH_GRID}"
[[ -n "${_SAVED_E_INIT_GRID}" ]] && export E_INIT_GRID="${_SAVED_E_INIT_GRID}"
[[ -n "${_SAVED_A_INIT_GRID}" ]] && export A_INIT_GRID="${_SAVED_A_INIT_GRID}"
[[ -n "${_SAVED_T_INIT_GRID}" ]] && export T_INIT_GRID="${_SAVED_T_INIT_GRID}"
[[ -n "${_SAVED_E_SCALE_MODE}" ]] && export E_SCALE_MODE="${_SAVED_E_SCALE_MODE}"
[[ -n "${_SAVED_LAMBDA_MECH_GRID}" ]] && export LAMBDA_MECH_GRID="${_SAVED_LAMBDA_MECH_GRID}"
[[ -n "${_SAVED_LAMBDA_TARGET_GRID}" ]] && export LAMBDA_TARGET_GRID="${_SAVED_LAMBDA_TARGET_GRID}"
[[ -n "${_SAVED_LAMBDA_TARGET}" ]] && export LAMBDA_TARGET="${_SAVED_LAMBDA_TARGET}"
[[ -n "${_SAVED_TARGET_NONE_POLICY}" ]] && export TARGET_NONE_POLICY="${_SAVED_TARGET_NONE_POLICY}"
[[ -n "${_SAVED_PROMPT_MODE}" ]] && export PROMPT_MODE="${_SAVED_PROMPT_MODE}"
[[ -n "${_SAVED_ENCODER_RUN_TAG}" ]] && export ENCODER_RUN_TAG="${_SAVED_ENCODER_RUN_TAG}"
[[ -n "${_SAVED_APPROACH}" ]] && export APPROACH="${_SAVED_APPROACH}"
[[ -n "${_SAVED_AMP_POOL}" ]] && export AMP_POOL="${_SAVED_AMP_POOL}"
[[ -n "${_SAVED_FUSION_LEVEL}" ]] && export FUSION_LEVEL="${_SAVED_FUSION_LEVEL}"
[[ -n "${_SAVED_TEACHER_OUTPUT_TAG}" ]] && export TEACHER_OUTPUT_TAG="${_SAVED_TEACHER_OUTPUT_TAG}"
[[ -n "${_SAVED_TOY_OUTPUT_TAG}" ]] && export TOY_OUTPUT_TAG="${_SAVED_TOY_OUTPUT_TAG}"
[[ -n "${_SAVED_TOY_HARD_TOKEN}" ]] && export TOY_HARD_TOKEN="${_SAVED_TOY_HARD_TOKEN}"
[[ -n "${_SAVED_ECHOHATE_DATA_SEED_DIR}" ]] && export ECHOHATE_DATA_SEED_DIR="${_SAVED_ECHOHATE_DATA_SEED_DIR}"
[[ -n "${_SAVED_DATASETS}" ]] && export DATASETS="${_SAVED_DATASETS}"
[[ -n "${_SAVED_HARD_ENCODERS}" ]] && export HARD_ENCODERS="${_SAVED_HARD_ENCODERS}"
[[ -n "${_SAVED_SKIP_ADAPT}" ]] && export SKIP_ADAPT="${_SAVED_SKIP_ADAPT}"
[[ -n "${_SAVED_TRACK_DYNAMICS}" ]] && export TRACK_DYNAMICS="${_SAVED_TRACK_DYNAMICS}"
[[ -n "${_SAVED_RANDOM_STATE}" ]] && export RANDOM_STATE="${_SAVED_RANDOM_STATE}"
[[ -n "${_SAVED_SEED}" ]] && export SEED="${_SAVED_SEED}"
[[ -n "${_SAVED_LLM_TEACHER_DIR}" ]] && export LLM_TEACHER_DIR="${_SAVED_LLM_TEACHER_DIR}"
[[ -n "${_SAVED_AMPVALTEST}" ]] && export AMPVALTEST="${_SAVED_AMPVALTEST}"
[[ -n "${_SAVED_LLM_CSV_VALIDATION}" ]] && export LLM_CSV_VALIDATION="${_SAVED_LLM_CSV_VALIDATION}"
[[ -n "${_SAVED_LLM_CSV_TEST}" ]] && export LLM_CSV_TEST="${_SAVED_LLM_CSV_TEST}"
[[ -n "${_SAVED_RESELECT_BEST_EPOCH}" ]] && export RESELECT_BEST_EPOCH="${_SAVED_RESELECT_BEST_EPOCH}"
[[ -n "${_SAVED_AMPVALTEST_OUTPUT_PREFIX}" ]] && export AMPVALTEST_OUTPUT_PREFIX="${_SAVED_AMPVALTEST_OUTPUT_PREFIX}"
[[ -n "${_SAVED_SPAN_OP}" ]] && export SPAN_OP="${_SAVED_SPAN_OP}"
[[ -n "${_SAVED_SPAN_CHANNELS}" ]] && export SPAN_CHANNELS="${_SAVED_SPAN_CHANNELS}"
[[ -n "${_SAVED_LLM_NORMAL_COL}" ]] && export LLM_NORMAL_COL="${_SAVED_LLM_NORMAL_COL}"
[[ -n "${_SAVED_SPAN_COVERAGE_MAX}" ]] && export SPAN_COVERAGE_MAX="${_SAVED_SPAN_COVERAGE_MAX}"
[[ -n "${_SAVED_ATTN_DEAMP_LAYERS}" ]] && export ATTN_DEAMP_LAYERS="${_SAVED_ATTN_DEAMP_LAYERS}"
[[ -n "${_SAVED_HATE_MECH_TAXONOMY}" ]] && export HATE_MECH_TAXONOMY="${_SAVED_HATE_MECH_TAXONOMY}"
[[ -n "${_SAVED_MECH_AUX_MODE}" ]] && export MECH_AUX_MODE="${_SAVED_MECH_AUX_MODE}"
[[ -n "${_SAVED_PACK_LAMBDA_GRID}" ]] && export PACK_LAMBDA_GRID="${_SAVED_PACK_LAMBDA_GRID}"
[[ -n "${_SAVED_PACK_SBATCH_TIME}" ]] && export PACK_SBATCH_TIME="${_SAVED_PACK_SBATCH_TIME}"

export ECHOHATE_DATA_SEED_DIR="${ECHOHATE_DATA_SEED_DIR:-eacl_seed0}"
export HATE_MECH_TAXONOMY=8way
echo "[from_local] HATE_MECH_TAXONOMY=${HATE_MECH_TAXONOMY}"
export DATASETS="${DATASETS:-HatEval}"
export HARD_ENCODERS="${HARD_ENCODERS:-all}"
export TEACHER_FAMILIES="${TEACHER_FAMILIES:-phi,qwen}"
export TEACHER_OUTPUT_TAG="${TEACHER_OUTPUT_TAG:-_eacl_batch8}"
export TOY_OUTPUT_TAG="${TOY_OUTPUT_TAG:-eacl}"
export TOY_HARD_TOKEN="${TOY_HARD_TOKEN:-hard}"
export TRACK_DYNAMICS="${TRACK_DYNAMICS:-1}"
export PROMPT_MODE="${PROMPT_MODE:-nocontext}"
export LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-8}"
export TEACHER_MODES="${TEACHER_MODES:-nocontext}"
# When adapting hard_prompt_GT, teachers live under llm_*_selected_hard_prompt_GT_*.
if [[ "${TOY_HARD_TOKEN}" == "hard_prompt_GT" ]]; then
  export LLM_SEL_HARD_MID="${LLM_SEL_HARD_MID:-selected_hard_prompt_GT}"
  # Keep GT mech runs separate from classic hard mech_only.
  if [[ "${ENCODER_RUN_TAG:-}" == "" || "${ENCODER_RUN_TAG}" == "_mech_only" ]]; then
    export ENCODER_RUN_TAG="_mech_only_gt"
  fi
else
  export LLM_SEL_HARD_MID="${LLM_SEL_HARD_MID:-selected_hard}"
fi
# Target-only: λ_mech=0, sweep λ_target (paper 11-grid unless caller sets LAMBDA_TARGET_GRID).
if [[ "${ENCODER_RUN_TAG}" == "_target_only" || "${TARGET_ONLY:-0}" == "1" ]]; then
  export APPROACH=mech
  export LAMBDA_MECH_GRID=0
  export LAMBDA_TARGET_GRID="${LAMBDA_TARGET_GRID:-0.01,0.03,0.05,0.07,0.09,0.1,0.3,0.5,0.7,0.9,1.0}"
  export TARGET_NONE_POLICY="${TARGET_NONE_POLICY:-mask}"
  export ENCODER_RUN_TAG="_target_only"
fi
# Default mech-only.
# APPROACH=mech only.
export APPROACH="${APPROACH:-mech}"
if [[ "${APPROACH}" == "cue" ]] && [[ "${ENCODER_RUN_TAG:-}" == "" || "${ENCODER_RUN_TAG}" == "_mech_only" ]]; then
  export ENCODER_RUN_TAG="_attn_only"
fi
export ENCODER_RUN_TAG="${ENCODER_RUN_TAG:-_mech_only}"
# Amplify attack-only / target-only: keep dirs distinct from union attn-only.
SPAN_CHANNELS="$(echo "${SPAN_CHANNELS:-both}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
case "${SPAN_CHANNELS}" in
  both|union|all|target_attack) SPAN_CHANNELS="both" ;;
  attack|attack_only|attack_cue|a) SPAN_CHANNELS="attack" ;;
  target|target_only|t) SPAN_CHANNELS="target" ;;
  *)
    echo "ERROR: SPAN_CHANNELS must be both|attack|target (got ${SPAN_CHANNELS})" >&2
    exit 1
    ;;
esac
export SPAN_CHANNELS
if [[ "${APPROACH}" == "cue" && "${SPAN_CHANNELS}" == "attack" ]]; then
  if [[ "${ENCODER_RUN_TAG}" == "_attn_only" || "${ENCODER_RUN_TAG}" == "_mech_only" ]]; then
    export ENCODER_RUN_TAG="_attn_only_attack"
  elif [[ "${ENCODER_RUN_TAG}" != *attack* ]]; then
    export ENCODER_RUN_TAG="${ENCODER_RUN_TAG}_attack"
  fi
elif [[ "${APPROACH}" == "cue" && "${SPAN_CHANNELS}" == "target" ]]; then
  if [[ "${ENCODER_RUN_TAG}" == "_attn_only" || "${ENCODER_RUN_TAG}" == "_mech_only" ]]; then
    export ENCODER_RUN_TAG="_attn_only_target"
  elif [[ "${ENCODER_RUN_TAG}" != *target* ]]; then
    export ENCODER_RUN_TAG="${ENCODER_RUN_TAG}_target"
  fi
fi
export FUSION_LEVEL="${FUSION_LEVEL:-post}"
# Early fusion: default tag if caller only set FUSION_LEVEL=early.
if [[ "${FUSION_LEVEL}" == "early" || "${FUSION_LEVEL}" == "input" ]]; then
  export FUSION_LEVEL=early
  if [[ "${ENCODER_RUN_TAG}" == "_attn_only" || "${ENCODER_RUN_TAG}" == "_attn_only_attack" || "${ENCODER_RUN_TAG}" == "_attn_only_target" || "${ENCODER_RUN_TAG}" == "_full_cue" || "${ENCODER_RUN_TAG}" == "_mech_only" ]]; then
    export ENCODER_RUN_TAG="_early_fusion"
  fi
fi
# Dual-view: span view + mechanism-conditioned view in one forward.
if [[ "${FUSION_LEVEL}" == "dual" || "${FUSION_LEVEL}" == "dual_view" || "${FUSION_LEVEL}" == "dualview" ]]; then
  export FUSION_LEVEL=dual_view
  if [[ "${ENCODER_RUN_TAG}" == "_attn_only" || "${ENCODER_RUN_TAG}" == "_attn_only_attack" || "${ENCODER_RUN_TAG}" == "_attn_only_target" || "${ENCODER_RUN_TAG}" == "_full_cue" || "${ENCODER_RUN_TAG}" == "_mech_only" || "${ENCODER_RUN_TAG}" == "_early_fusion" ]]; then
    export ENCODER_RUN_TAG="_dual_view"
  fi
fi
export GRID_PRESET="${GRID_PRESET:-full}"
# Mech-only λ grid — same 4 values as default full cue grid (CUE_LAMBDA_MECH_GRID).
_DEFAULT_GRID="0.01,0.05,0.1,0.5"
export LAMBDA_MECH_GRID="${LAMBDA_MECH_GRID:-${_DEFAULT_GRID}}"
if [[ "${APPROACH}" == "cue" ]]; then
  export E_INIT_GRID="${E_INIT_GRID:-${_DEFAULT_GRID}}"
  # Only default λ→0 for attn-only tags. Full-cue / custom tags keep/use the 4-λ grid.
  if [[ "${ENCODER_RUN_TAG}" == "_attn_only" || "${ENCODER_RUN_TAG}" == "_attn_only_at" || "${ENCODER_RUN_TAG}" == "_attn_only_attack" || "${ENCODER_RUN_TAG}" == "_attn_only_target" ]]; then
    export CUE_LAMBDA_MECH_GRID="${CUE_LAMBDA_MECH_GRID:-0}"
  else
    export CUE_LAMBDA_MECH_GRID="${CUE_LAMBDA_MECH_GRID:-${_DEFAULT_GRID}}"
  fi
  if [[ "${ENCODER_RUN_TAG}" == "_attn_only_at" || "${E_SCALE_MODE:-}" == "split" ]]; then
    export E_SCALE_MODE=split
    export A_INIT_GRID="${A_INIT_GRID:-${_DEFAULT_GRID}}"
    export T_INIT_GRID="${T_INIT_GRID:-${_DEFAULT_GRID}}"
  fi
fi
export ADAPT_ONLY="${ADAPT_ONLY:-0}"
export SKIP_ADAPT="${SKIP_ADAPT:-0}"
export SYNC_TO_HPC="${SYNC_TO_HPC:-1}"
export SPLIT_SUBDIR="${SPLIT_SUBDIR:-splitted_data_80_10_10}"
export CONDA_ENV="${CONDA_ENV:-rag_v8_gpu}"
# eacl_seed0 is not seed_N_model → post_llm would not auto-align; force 0.
# Prefer caller RANDOM_STATE; else infer from eacl_seedN_seq / seed_N*; else 0.
if [[ -z "${RANDOM_STATE:-}" ]]; then
  if [[ "${ECHOHATE_DATA_SEED_DIR:-}" =~ eacl_seed([0-9]+)(_|$) ]]; then
    RANDOM_STATE="${BASH_REMATCH[1]}"
  elif [[ "${ECHOHATE_DATA_SEED_DIR:-}" =~ ^seed_([0-9]+) ]]; then
    RANDOM_STATE="${BASH_REMATCH[1]}"
  else
    RANDOM_STATE=0
  fi
fi
export RANDOM_STATE
export RANDOM_STATE_OVERRIDE="${RANDOM_STATE_OVERRIDE:-1}"
# Disable bert+phi remap HatEval→_new_prompt (would miss llm_*_eacl_batch8 dirs).
export BERT_PHI_NEW_PROMPT_ONLY_DATASETS="${BERT_PHI_NEW_PROMPT_ONLY_DATASETS:-}"

HPC_DIR="${HPC_RA}/experiment/echohate"
DATA_ROOT_REL="data/${ECHOHATE_DATA_SEED_DIR}"
export ENCODER_DATA_ROOT="${HPC_DIR}/data/${ECHOHATE_DATA_SEED_DIR}"
LOCAL_ECHOHATE="$(cd "${_BOOT}/../../.." && pwd)"
MECH_SUBMIT="${_BOOT}/submit_roberta_fft_mech_cue_hpclogin_dgx_from_local.sh"
ADAPT_PY="${LOCAL_ECHOHATE}/pipeline/stage1/adapt_toy_prompt_hard_for_mech.py"

if [[ ! -f "${MECH_SUBMIT}" ]]; then
  echo "ERROR: missing ${MECH_SUBMIT}" >&2
  exit 1
fi
if [[ ! -f "${ADAPT_PY}" ]]; then
  echo "ERROR: missing ${ADAPT_PY}" >&2
  echo "The LLM teacher adapter is not in this partial release. Train with:" >&2
  echo "  python -m pipeline stage2 mech_distill --accepted_selected_k_csv <mechanism_labels.csv>" >&2
  echo "The teacher and deferral code will be released upon acceptance." >&2
  exit 1
fi

echohate_export_hpc_password_from_secrets "${LOCAL_ECHOHATE}"
export HPC_SSHPASS_FOR_MKDIR="${HPC_SSHPASS_FOR_MKDIR:-1}"
export HPC_SSH_SUBMIT_MUX="${HPC_SSH_SUBMIT_MUX:-1}"

IFS=',' read -r -a _DATASET_LIST <<< "${DATASETS}"
IFS=',' read -r -a _FAM_LIST <<< "${TEACHER_FAMILIES}"

_teacher_tag="${TEACHER_OUTPUT_TAG}"
if [[ -n "${_teacher_tag}" && "${_teacher_tag}" != _* ]]; then
  _teacher_tag="_${_teacher_tag}"
fi

_discover_encoders() {
  local ds="$1"
  local root="${LOCAL_ECHOHATE}/${DATA_ROOT_REL}/${ds}"
  local d name
  shopt -s nullglob
  for d in "${root}"/*/selection_gmm/selected_hard_rows_for_inference.csv; do
    name="$(basename "$(dirname "$(dirname "${d}")")")"
    printf '%s\n' "${name}"
  done
  shopt -u nullglob
}

_encoders_for_ds() {
  local ds="$1"
  if [[ "${HARD_ENCODERS}" == "all" ]]; then
    _discover_encoders "${ds}"
    return
  fi
  local enc
  IFS=',' read -r -a _enc_arr <<< "${HARD_ENCODERS}"
  for enc in "${_enc_arr[@]}"; do
    enc="$(echo "${enc}" | xargs)"
    [[ -n "${enc}" ]] && printf '%s\n' "${enc}"
  done
}

_student_from_encoder() {
  case "$1" in
    bert*) echo "bert-large-uncased bert_large" ;;
    roberta*) echo "roberta-large roberta_large" ;;
    *)
      echo "ERROR: cannot infer student model from encoder name: $1" >&2
      return 1
      ;;
  esac
}

_ce_from_encoder() {
  case "$1" in
    *_woCL) echo "none" ;;
    *_wCL) echo "0.5" ;;
    *)
      if [[ -n "${CE_CONTRASTIVE_LAMBDA:-}" ]]; then
        echo "${CE_CONTRASTIVE_LAMBDA}"
      else
        echo "none"
      fi
      ;;
  esac
}

_set_encoder_baseline_rel() {
  local ds="$1"
  local rel="$2"
  case "${ds}" in
    IHC) export ENCODER_IHC_BASELINE_RUN_REL="${rel}" ;;
    DynaHate) export ENCODER_DYNAHATE_BASELINE_RUN_REL="${rel}" ;;
    HatEval) export ENCODER_HATEVAL_BASELINE_RUN_REL="${rel}" ;;
    HateXplain) export ENCODER_HATEXPLAIN_BASELINE_RUN_REL="${rel}" ;;
    Ethos) export ENCODER_ETHOS_BASELINE_RUN_REL="${rel}" ;;
    offensive_as_hate) export ENCODER_OFFENSIVE_AS_HATE_BASELINE_RUN_REL="${rel}" ;;
    offensive_as_not_hate) export ENCODER_OFFENSIVE_AS_NOT_HATE_BASELINE_RUN_REL="${rel}" ;;
    offensive_as_not_hate_soft_none) export ENCODER_OFFENSIVE_AS_NOT_HATE_SOFT_NONE_BASELINE_RUN_REL="${rel}" ;;
    drop_offensive) export ENCODER_DROP_OFFENSIVE_BASELINE_RUN_REL="${rel}" ;;
    drop_offensive_repo_split) export ENCODER_DROP_OFFENSIVE_REPO_SPLIT_BASELINE_RUN_REL="${rel}" ;;
    SBIC) export ENCODER_SBIC_BASELINE_RUN_REL="${rel}" ;;
    ToxiGen|toxigen_new) export ENCODER_TOXIGEN_BASELINE_RUN_REL="${rel}" ;;
    white_new) export ENCODER_WHITE_NEW_BASELINE_RUN_REL="${rel}" ;;
    *)
      echo "WARN: no ENCODER_* slot for dataset=${ds}; baseline rel=${rel}" >&2
      ;;
  esac
}

_run_adapt() {
  local ds="$1"
  local enc="$2"
  PYTHONPATH="${LOCAL_ECHOHATE}${PYTHONPATH:+:${PYTHONPATH}}" python3 "${ADAPT_PY}" \
    --data-root "${LOCAL_ECHOHATE}/${DATA_ROOT_REL}" \
    --dataset "${ds}" \
    --encoders "${enc}" \
    --prompt-mode "${PROMPT_MODE}" \
    --teacher-families "${TEACHER_FAMILIES}" \
    --teacher-output-tag "${_teacher_tag}" \
    --toy-output-tag "${TOY_OUTPUT_TAG}" \
    --hard-token "${TOY_HARD_TOKEN}" \
    --llm-batch-size "${LLM_BATCH_SIZE}"
}

_sync_encoder_wave() {
  local ds="$1"
  local enc="$2"
  local rel="${DATA_ROOT_REL}/${ds}/${enc}"
  echo "Sync → HPC: ${rel}/ (selection_variability_top25/selection_gmm, splits, llm_*_selected_hard_*)"
  hpc_mkdir_remote_from_local \
    "${HPC_DIR}/${DATA_ROOT_REL}/${ds}/${enc}" \
    "${HPC_DIR}/${DATA_ROOT_REL}/${ds}/${SPLIT_SUBDIR}"
  if [[ -d "${LOCAL_ECHOHATE}/${rel}/selection_variability_top25" ]]; then
    hpc_rsync_with_retry -avz \
      "${LOCAL_ECHOHATE}/${rel}/selection_variability_top25/" \
      "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/${rel}/selection_variability_top25/"
  fi
  if [[ -d "${LOCAL_ECHOHATE}/${rel}/selection_variability_top25_cross_union" ]]; then
    hpc_rsync_with_retry -avz \
      "${LOCAL_ECHOHATE}/${rel}/selection_variability_top25_cross_union/" \
      "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/${rel}/selection_variability_top25_cross_union/"
  fi
  if [[ -d "${LOCAL_ECHOHATE}/${rel}/selection_gmm" ]]; then
    hpc_rsync_with_retry -avz \
      "${LOCAL_ECHOHATE}/${rel}/selection_gmm/" \
      "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/${rel}/selection_gmm/"
  fi
  hpc_rsync_with_retry -avz \
    "${LOCAL_ECHOHATE}/${DATA_ROOT_REL}/${ds}/${SPLIT_SUBDIR}/" \
    "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/${DATA_ROOT_REL}/${ds}/${SPLIT_SUBDIR}/"
  for fam in "${_FAM_LIST[@]}"; do
    fam="$(echo "${fam}" | xargs)"
    [[ -z "${fam}" ]] && continue
    local _mid="${LLM_SEL_HARD_MID:-selected_hard}"
    local tdir="${LOCAL_ECHOHATE}/${rel}/llm_${fam}_${_mid}_nocontext${_teacher_tag}"
    if [[ -d "${tdir}" ]]; then
      hpc_rsync_with_retry -avz \
        "${tdir}/" \
        "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/${rel}/llm_${fam}_${_mid}_nocontext${_teacher_tag}/"
    fi
  done
}

echo "=========================================="
echo "mech-only from toy_prompt hard"
echo "  seed:      ${ECHOHATE_DATA_SEED_DIR}"
echo "  datasets:  ${_DATASET_LIST[*]}"
echo "  encoders:  ${HARD_ENCODERS}"
echo "  teachers:  ${_FAM_LIST[*]}  tag=${_teacher_tag}"
echo "  hard:      TOY_HARD_TOKEN=${TOY_HARD_TOKEN}  LLM_SEL_HARD_MID=${LLM_SEL_HARD_MID}"
echo "  modes:     ${TEACHER_MODES}  ENCODER_RUN_TAG=${ENCODER_RUN_TAG}"
echo "  dynamics:  TRACK_DYNAMICS=${TRACK_DYNAMICS:-1}"
echo "  adapt:     SKIP_ADAPT=${SKIP_ADAPT}  ADAPT_ONLY=${ADAPT_ONLY}"
echo "  sync HPC:  ${SYNC_TO_HPC}"
echo "  HPC:       ${HPC_USER}@${HPC_HOST}:${HPC_DIR}"
echo "=========================================="

_submitted=0
for ds in "${_DATASET_LIST[@]}"; do
  ds="$(echo "${ds}" | xargs)"
  [[ -z "${ds}" ]] && continue

  _encs=()
  while IFS= read -r _line; do
    _line="$(echo "${_line}" | xargs)"
    [[ -n "${_line}" ]] && _encs+=("${_line}")
  done < <(_encoders_for_ds "${ds}")

  if [[ ${#_encs[@]} -eq 0 ]]; then
    echo "ERROR: no encoders under ${DATA_ROOT_REL}/${ds}/" >&2
    exit 1
  fi

  for enc in "${_encs[@]}"; do
    echo ""
    echo ">>> ${ds} / ${enc}"

    if [[ "${SKIP_ADAPT}" != "1" ]]; then
      _run_adapt "${ds}" "${enc}"
    fi

    if [[ "${SYNC_TO_HPC}" == "1" ]]; then
      _sync_encoder_wave "${ds}" "${enc}"
    fi

    if [[ "${ADAPT_ONLY}" == "1" ]]; then
      continue
    fi

    read -r _student _bn <<< "$(_student_from_encoder "${enc}")"
    _ce="$(_ce_from_encoder "${enc}")"
    _baseline_rel="${DATA_ROOT_REL}/${ds}/${enc}"

    for fam in "${_FAM_LIST[@]}"; do
      fam="$(echo "${fam}" | xargs)"
      [[ -z "${fam}" ]] && continue

      # Fresh exports per wave (avoid stale TEACHER_* / STUDENT_* leaking).
      unset STUDENT_MODEL_NAME VALIDATION_ROOT_ENCODER_BN TEACHER_LM_FAMILY \
        TEACHER_OUTPUT_TAG TEACHER_OUTPUT_TAG_OVERRIDES CE_CONTRASTIVE_LAMBDA \
        TEACHER_OUTPUT_TAG_PHI TEACHER_OUTPUT_TAG_QWEN TEACHER_OUTPUT_TAG_LLAMA \
        TEACHER_OUTPUT_TAG_OVERRIDES_PHI TEACHER_OUTPUT_TAG_OVERRIDES_QWEN \
        TEACHER_OUTPUT_TAG_OVERRIDES_LLAMA ENCODER_OUT_DIR_BN \
        ENCODER_IHC_BASELINE_RUN_REL ENCODER_DYNAHATE_BASELINE_RUN_REL \
        ENCODER_HATEVAL_BASELINE_RUN_REL ENCODER_HATEXPLAIN_BASELINE_RUN_REL \
        ENCODER_ETHOS_BASELINE_RUN_REL ENCODER_OFFENSIVE_AS_HATE_BASELINE_RUN_REL \
        ENCODER_OFFENSIVE_AS_NOT_HATE_BASELINE_RUN_REL \
        ENCODER_OFFENSIVE_AS_NOT_HATE_SOFT_NONE_BASELINE_RUN_REL \
        ENCODER_DROP_OFFENSIVE_BASELINE_RUN_REL ENCODER_DROP_OFFENSIVE_REPO_SPLIT_BASELINE_RUN_REL \
        ENCODER_SBIC_BASELINE_RUN_REL ENCODER_TOXIGEN_BASELINE_RUN_REL \
        ENCODER_WHITE_NEW_BASELINE_RUN_REL

      export TOY_PROMPT_HARD_TEACHER=1

      export CE_CONTRASTIVE_LAMBDA_EXPLICIT=1
      export CE_CONTRASTIVE_LAMBDA="${_ce}"
      export ECHOHATE_DATA_SEED_DIR
      export ONLY_DATASETS="${ds}"
      export APPROACH
      export ENCODER_RUN_TAG
      export TEACHER_MODES
      export TEACHER_LM_FAMILY="${fam}"
      export TEACHER_OUTPUT_TAG="${_teacher_tag}"
      export TEACHER_OUTPUT_TAG_OVERRIDES=""
      export BERT_PHI_NEW_PROMPT_ONLY_DATASETS=""
      export LLM_SEL_HARD_LAYOUT=phi_first
      export LLM_SEL_HARD_MID="${LLM_SEL_HARD_MID:-selected_hard}"
      export STUDENT_MODEL_NAME="${_student}"
      export VALIDATION_ROOT_ENCODER_BN="${_bn}"
      # Keep wCL/woCL separate under eacl_seed0 (same seed bucket hosts both baselines).
      # → …/HatEval/roberta_large_baseline_wCL_encoder/encoder_fft_{mech,cue_span}_…
      export ENCODER_OUT_DIR_BN="${enc}"
      export GRID_PRESET
      export LAMBDA_MECH_GRID
      export LAMBDA_TARGET_GRID="${LAMBDA_TARGET_GRID:-0}"
      export LAMBDA_TARGET_RAMP_RATIO="${LAMBDA_TARGET_RAMP_RATIO:-0.1}"
      export TARGET_NONE_POLICY="${TARGET_NONE_POLICY:-mask}"
      if [[ "${APPROACH}" == "cue" ]]; then
        export CUE_LAMBDA_MECH_GRID="${CUE_LAMBDA_MECH_GRID:-0}"
        export E_INIT_GRID="${E_INIT_GRID:-${_DEFAULT_GRID}}"
        export AMP_POOL="${AMP_POOL:-sum}"
        export FUSION_LEVEL="${FUSION_LEVEL:-post}"
      fi
      export TRACK_DYNAMICS="${TRACK_DYNAMICS:-1}"
      export CONDA_ENV
      export RANDOM_STATE
      export RANDOM_STATE_OVERRIDE=1
      export ENCODER_DATA_ROOT
      export HPC_SKIP_CODE_RSYNC="${HPC_SKIP_CODE_RSYNC:-0}"

      # Re-apply overrides that survive fam-loop unset / ambient wipe.
      if [[ -n "${_SAVED_LLM_TEACHER_DIR}" ]]; then
        export LLM_TEACHER_DIR="${_SAVED_LLM_TEACHER_DIR}"
      fi
      if [[ -n "${_SAVED_SPAN_OP}" ]]; then
        export SPAN_OP="${_SAVED_SPAN_OP}"
      fi
      if [[ -n "${_SAVED_SPAN_CHANNELS}" ]]; then
        export SPAN_CHANNELS="${_SAVED_SPAN_CHANNELS}"
      fi
      if [[ -n "${_SAVED_LLM_NORMAL_COL}" ]]; then
        export LLM_NORMAL_COL="${_SAVED_LLM_NORMAL_COL}"
      fi
      if [[ -n "${_SAVED_SPAN_COVERAGE_MAX}" ]]; then
        export SPAN_COVERAGE_MAX="${_SAVED_SPAN_COVERAGE_MAX}"
      fi
      if [[ -n "${_SAVED_ATTN_DEAMP_LAYERS}" ]]; then
        export ATTN_DEAMP_LAYERS="${_SAVED_ATTN_DEAMP_LAYERS}"
      fi
      if [[ -n "${_SAVED_AMPVALTEST}" ]]; then
        export AMPVALTEST="${_SAVED_AMPVALTEST}"
      fi
      if [[ -n "${_SAVED_LLM_CSV_VALIDATION}" ]]; then
        export LLM_CSV_VALIDATION="${_SAVED_LLM_CSV_VALIDATION}"
      fi
      if [[ -n "${_SAVED_LLM_CSV_TEST}" ]]; then
        export LLM_CSV_TEST="${_SAVED_LLM_CSV_TEST}"
      fi
      if [[ -n "${_SAVED_RESELECT_BEST_EPOCH}" ]]; then
        export RESELECT_BEST_EPOCH="${_SAVED_RESELECT_BEST_EPOCH}"
      fi
      if [[ -n "${_SAVED_AMPVALTEST_OUTPUT_PREFIX}" ]]; then
        export AMPVALTEST_OUTPUT_PREFIX="${_SAVED_AMPVALTEST_OUTPUT_PREFIX}"
      fi

      _set_encoder_baseline_rel "${ds}" "${_baseline_rel}"

      echo "SUBMIT ${APPROACH} ${ds} enc=${enc} student=${_student} ce=${_ce} teacher=${fam} tag=${ENCODER_RUN_TAG}"
      if [[ "${APPROACH}" == "cue" ]]; then
        echo "  TEACHER_OUTPUT_TAG=${TEACHER_OUTPUT_TAG}  CUE_LAMBDA_MECH_GRID=${CUE_LAMBDA_MECH_GRID}  E_INIT_GRID=${E_INIT_GRID}  AMP_POOL=${AMP_POOL}  FUSION_LEVEL=${FUSION_LEVEL}  RANDOM_STATE=${RANDOM_STATE}"
        if [[ -n "${LLM_TEACHER_DIR:-}" ]]; then
          echo "  LLM_TEACHER_DIR=${LLM_TEACHER_DIR}"
        fi
        if [[ "${AMPVALTEST:-0}" == "1" ]]; then
          echo "  AMPVALTEST=1  LLM_CSV_VALIDATION=${LLM_CSV_VALIDATION:-?}  LLM_CSV_TEST=${LLM_CSV_TEST:-?}"
        fi
      else
        echo "  TEACHER_OUTPUT_TAG=${TEACHER_OUTPUT_TAG}  LAMBDA_MECH_GRID=${LAMBDA_MECH_GRID}  LAMBDA_TARGET_GRID=${LAMBDA_TARGET_GRID:-0}  TARGET_NONE_POLICY=${TARGET_NONE_POLICY:-mask}  RANDOM_STATE=${RANDOM_STATE}  DATA_ROOT=${ENCODER_DATA_ROOT}  LLM_SEL_HARD_MID=${LLM_SEL_HARD_MID}"
      fi
      echo "  OUT=…/${ds}/${ENCODER_OUT_DIR_BN}_encoder/  TEACHER=…/${ds}/${enc}/llm_${fam}_${LLM_SEL_HARD_MID}_*"
      bash "${MECH_SUBMIT}"
      _submitted=$((_submitted + 1))
    done
  done
done

echo "=========================================="
echo "Done. submit_waves=${_submitted}  (each wave = grid for one ds×encoder×teacher; APPROACH=${APPROACH})"
if [[ "${ADAPT_ONLY}" == "1" ]]; then
  echo "ADAPT_ONLY=1 — no sbatch."
fi
echo "Track: ssh ${HPC_USER}@${HPC_HOST} 'squeue -u ${HPC_USER}'"
echo "=========================================="
