#!/usr/bin/env bash
# Require bash (do not run as `sh script.sh` — POSIX sh will mis-parse and fail late with cryptic errors).
if [ -z "${BASH_VERSION:-}" ]; then
  echo "ERROR: run this script with bash, e.g.  bash \"$0\"  (not sh \"$0\")." >&2
  exit 1
fi
# Submit RoBERTa-large **full fine-tune** after selected-hard LLM (mech-only):
#   mech_distill.py — CE + mechanism auxiliary on agreed train/validation rows
#   APPROACH=mech only
#
#   Mech: APPROACH=mech  ENCODER_RUN_TAG=_mech_only
#     → slurm_mech_distill_single.sh, LAMBDA_MECH grid (default: 1 sbatch / λ)
#     PACK_LAMBDA_GRID=1 → 1 sbatch / teacher×mode loops all λ sequentially
#
# Prereq on HPC: under each baseline run dir (e.g. data/seed_0/<DS>/roberta_large_baseline/),
#   llm_selected_hard_phi_* folders with selected_k_llm_predictions.csv (nocontext / traincontext).
#
# Usage (CLI flags — no export required):
#   cd .../echohate
#   bash job_submission/stage2/cue_mech_attnmask/submit_roberta_fft_mech_cue_post_llm_from_local.sh \
#     --seed seed_0 --dataset IHC --approach mech --teacher-modes nocontext
#
# Positional shorthand (same flags):
#   bash .../submit_roberta_fft_mech_cue_post_llm_from_local.sh seed_0 IHC --approach mech --teacher-modes nocontext
#
# Env vars still work if already set in the shell; flags override unset defaults.
#
# Legacy hpclogin + DGX partition (instead of hpc2 + g46-*):
#   bash job_submission/stage2/cue_mech_attnmask/submit_roberta_fft_mech_cue_hpclogin_dgx_from_local.sh
#   (or export ECHOHATE_HPC_PROFILE=hpclogin_dgx before this script — see hpc_defaults_from_local.sh)
#
# Seed-bucket + custom encoder run dir (Phi llm + selection_gmm live under RUN_DIR):
#   export ENCODER_IHC_BASELINE_RUN_REL='data/seed_0/IHC/roberta_large_baseline'
#   export ENCODER_DYNAHATE_BASELINE_RUN_REL='data/seed_0/DynaHate/roberta_large_baseline'
#   export ENCODER_HATEVAL_BASELINE_RUN_REL='data/seed_0/HatEval/roberta_large_baseline'
#   Slurm DATA_ROOT auto-defaults to .../data/seed_<n> when a baseline path starts with data/seed_<n>/ (else set ENCODER_DATA_ROOT).
#
# Per-dataset preset submit_* wrappers were removed; use ONLY_DATASETS / CUE_MODE / ENCODER_RUN_TAG / grids in this Env section.
#
# Env:
#   **Unset before export** when switching encoder/llm waves in the same shell (stale exports otherwise win silently):
#     unset STUDENT_MODEL_NAME VALIDATION_ROOT_ENCODER_BN TEACHER_LM_FAMILY TEACHER_OUTPUT_TAG TEACHER_OUTPUT_TAG_OVERRIDES
#     unset TEACHER_OUTPUT_TAG_PHI TEACHER_OUTPUT_TAG_QWEN TEACHER_OUTPUT_TAG_LLAMA
#     unset TEACHER_OUTPUT_TAG_OVERRIDES_PHI TEACHER_OUTPUT_TAG_OVERRIDES_QWEN TEACHER_OUTPUT_TAG_OVERRIDES_LLAMA
#     then export the intended STUDENT_* / TEACHER_* values for this submit.
#   APPROACH=mech only
#   TEACHER_MODES=both|nocontext|traincontext (default: both)
#   TEACHER_MODES_ORDER=interleaved|nocontext_first
#   ONLY_DATASETS=csv e.g. DynaHate,HatEval  (empty = default RUN_SPECS for seed bucket)
#   ONLY_ETHOS_VARIANT=both|posfreq|posnone  (default both; limits Ethos posfreq vs posnone trees when ONLY_DATASETS includes Ethos)
#   GRID_PRESET=full|tiny  (default: full 4×4=16; tiny = small/moderate λ_mech and e_init for pilots)
#   LAMBDA_MECH_GRID=...  (mech / APPROACH=both only — **ignored when APPROACH=cue**)
#   PACK_LAMBDA_GRID=0|1  (mech only; default 0 = one sbatch per λ; 1 = one sbatch packs the λ list)
#   PACK_SBATCH_TIME=…    (optional walltime when PACK_LAMBDA_GRID=1; default scales 24h × n_λ, cap 72h)
#   CUE_LAMBDA_MECH_GRID=...  (cue: Cartesian with E_INIT_GRID; default **4×4 = 16** jobs per dataset×mode when unset)
#   E_INIT_GRID=...       (cue: default 4 e_init values; tiny preset uses 4; mech-only runs ignore E_INIT for sbatch count)
#   CE_CONTRASTIVE_LAMBDA=   (*_wCL → 0.5, *_woCL → none/CE-only; folder ALWAYS wins for those
#     buckets unless CE_CONTRASTIVE_LAMBDA_EXPLICIT=1. Export uses "none" not empty for Slurm safety.
#     other seeds → CE-only/"none" unless you export explicitly)
#   CONTRASTIVE_MARGIN=0.75  (logit cosine contrastive only)
# encoder jobs use a BERT/RoBERTa-family encoder (default roberta-large, full fine-tune).
#   RANDOM_STATE=0
#   Default: do **not** skip existing outputs. Opt-in: --skip-if-output-exists
#   (unset any export SKIP_IF_OUTPUT_EXISTS=1 in your shell — it is no longer read)
#   CHECK_TEACHER_AT_SUBMIT=1  (optional: verify llm CSV on HPC before submit; prefer *_trainid.csv when present)
#   CHECK_INPUTS_AT_SUBMIT=1|0  (default 1: ssh to HPC before sbatch per dataset×llm_mode — require llm CSV,
#     data/<DATASET>/<split>/train.csv, data/<DATASET>/<split>/validation.csv, and baseline_run RUN_DIR to exist;
#     skip that mode if anything is missing)
#   CHECK_INPUTS_STRICT=1  (exit 1 on first missing input instead of skipping that mode)
#   TEACHER_OUTPUT_TAG=  suffix appended in llm_selected_hard_phi_nocontext<HERE> / ..._traincontext<HERE>
#     on disk under each baseline_run RUN_DIR. **Not the same for every dataset or every Phi wave** — it is whatever
#     `slurm_llm_infer_on_hard.sh` produced: OUTPUT_TAG_SUFFIX (e.g. _new_prompt) + optional model slug (e.g.
#     _microsoft_phi-4 when PHI_SELHARD_OUTPUT_MODEL_TAG=1) + optional PHI_SELHARD_TRAILING_SUFFIX (e.g. _same_abstract).
#     Examples for Phi-4: `_new_prompt` only (legacy, no model in dirname); `_new_prompt_microsoft_phi-4`;
#     `_new_prompt_microsoft_phi-4_same_abstract`. **Qwen / Llama** selected-hard dirs follow the same slurm naming:
#     model slug (slashes → underscores) and often the same **`_same_abstract`** tail when that wave was run — include
#     it in TEACHER_OUTPUT_TAG / TEACHER_OUTPUT_TAG_QWEN / TEACHER_OUTPUT_TAG_LLAMA / overrides (e.g. `…_Qwen_…_same_abstract`).
#     Other HF ids change the slug. Always `ls` one baseline_run tree on the cluster and match the folder tail after
#     `llm_selected_hard_phi_nocontext`.
#   TEACHER_OUTPUT_TAG_OVERRIDES=  optional comma list `KEY=value,...` to set a different tag per row (recommended when
#     Phi dirs differ by dataset). Lookup order per RUN_SPEC row: **JOB_DS_TAG** first (e.g. Ethos_posfreq, IHC),
#     then bare **DATASET** name. Examples:
#       TEACHER_OUTPUT_TAG_OVERRIDES='HateXplain=_new_prompt_microsoft_phi-4_same_abstract,DynaHate=_new_prompt'
#       (HateXplain 811 BERT Phi traincontext/nocontext often use the `_same_abstract` tail; IHC/DynaHate/HatEval may use `_new_prompt` only.)
#     If no key matches, TEACHER_OUTPUT_TAG is used as fallback (may be empty for legacy unsuffixed dirs).
#   TEACHER_LM_FAMILY=phi|qwen|llama  (optional). **phi** + **bert-large** (`VALIDATION_ROOT_ENCODER_BN=bert_large`):
#     auto-appends TEACHER_OUTPUT_TAG_OVERRIDES for IHC,DynaHate,HatEval → `_new_prompt` only (Phi dirs without
#     LM tag there); other datasets still use TEACHER_OUTPUT_TAG (e.g. `_new_prompt_microsoft_phi-4` on 811 BERT Phi).
#     **qwen / llama**: set the full LM-tagged suffix from `ls` (often ends in `_same_abstract` like Phi); no bert_large trim.
#   BERT_PHI_NEW_PROMPT_ONLY_DATASETS=csv  (default IHC,DynaHate,HatEval; bert_large + phi only)
#   BERT_PHI_NEW_PROMPT_TAG=_new_prompt  (suffix for those rows; default _new_prompt)
#   One run = one encoder encoder tree (bert_large vs roberta_large). BERT Phi submit does **not** resubmit RoBERTa Phi;
#     run STUDENT_MODEL_NAME=roberta-large again only when you intentionally train RoBERTa encoders.
#   SUBMIT_STUDENT_TEACHER_PROFILES=  optional comma list `encoder_bn|lm_family,...` to walk **multiple** encoder+llm
#     waves in one script (sync once, then sbatch each wave). encoder_bn: bert_large|roberta_large|bert_base|roberta_base.
#     lm_family: phi|qwen|llama (llm dirs stay `llm_selected_hard_phi_*` on disk; TEACHER_OUTPUT_TAG carries the LM slug).
#     Example (BERT × three llms + RoBERTa × Qwen/Llama, no RoBERTa×Phi):
#       SUBMIT_STUDENT_TEACHER_PROFILES='bert_large|phi,bert_large|qwen,bert_large|llama,roberta_large|qwen,roberta_large|llama'
#     Per-family dirname suffixes from `ls baseline_run/` (include `_same_abstract` in the string when that wave exists):
#       TEACHER_OUTPUT_TAG_PHI TEACHER_OUTPUT_TAG_QWEN TEACHER_OUTPUT_TAG_LLAMA
#     Optional overrides per family: TEACHER_OUTPUT_TAG_OVERRIDES_PHI, TEACHER_OUTPUT_TAG_OVERRIDES_QWEN, TEACHER_OUTPUT_TAG_OVERRIDES_LLAMA (else shared TEACHER_OUTPUT_TAG_OVERRIDES).
#     Shared fallbacks when a per-family TEACHER_OUTPUT_TAG_* var is unset: TEACHER_OUTPUT_TAG, TEACHER_OUTPUT_TAG_OVERRIDES (as today).
#   INIT_BACKBONE_FROM= optional HPC path to FFT ``model/pytorch_model.pt``. Prefer:
#       ENCODER_RUN_TAG=_warmft  (or INIT_BACKBONE_AUTO=1) — submit auto-probes
#       ``<baseline_run>/model/pytorch_model.pt`` and requires a ``warmft`` dirname suffix so
#       warm-start grids do not collide with from-scratch cue/mech dirs. Loads backbone + classifier
#       into cue and mech students (``e`` / mechanism head stay at init). Do not set this for
#       multi-dataset submits — use ENCODER_RUN_TAG=_warmft so each DS probes its own baseline.
#   INIT_BACKBONE_AUTO=1  — same as tagging warmft: probe baseline pt + ensure ENCODER_RUN_TAG contains warmft.
#   INIT_LOAD_CLASSIFIER=0  — warm-start backbone only (default 1 = also copy Linear classifier head).
#   LLM_TEACHER_DIR= optional path (HPC abs or repo-relative) to a folder with
#       selected_k_llm_predictions.csv — bypasses run_root/llm_* discovery (mech→attn warmft).
#       May include ``{ds}`` (expanded to the current dataset name per submit row).
#   SPAN_OP=amplify|deamplify|attn_deamp  (cue jobs; default amplify). Must be distinct
#       ENCODER_RUN_TAG when comparing ops (output dirs do not embed span_op).
#   SPAN_CHANNELS=both|attack|target  (amplify only; default both). Attack-only → use
#       ENCODER_RUN_TAG=_attn_only_attack (auto-appended from _attn_only when SPAN_CHANNELS=attack).
#   LLM_NORMAL_COL=normal_spans  DEAMP_AUTO=1  (deamp: auto layers/coverage/mask_dropout; override with DEAMP_AUTO=0)
#   ENCODER_RUN_TAG= e.g. _attnmask | _warmft | _attn_only | _attn_only_attack | _deamp_residual
#       (extra suffix on encoder out dirs + job names; cue uses attention_mask_mode only).
#   GRADIENT_ACCUMULATION_STEPS=1  (cue jobs only; effective batch ≈ BATCH_SIZE × this)
#   GRADIENT_CHECKPOINTING=0|1  (cue: encoder activation checkpointing, lower VRAM)
#   RUN_EVAL_OVERWRITE_OUTPUTS=0  (default **1**: overwrite stale final_test_* after re-runs)
#   FORCE_RETRAIN=1  (default 0: skip train only when cue_span_state.pt + tokenizer exist AND
#     n_mechanism_classes matches code — currently 13 = hate8 + offensive5; stale 8-class SBIC
#     checkpoints auto-retrain even with FORCE_RETRAIN=0)
#   STUDENT_MODEL_NAME=roberta-large  (optional; **wins** over MODEL_NAME — use for encoder encoder)
#     Default encoder is roberta-large full FT. Stale MODEL_NAME=microsoft/phi-4 from llm shells is ignored.
#   VALIDATION_ROOT_ENCODER_BN=roberta_large|bert_large  (optional; default inferred from STUDENT_MODEL_NAME /
#     MODEL_NAME so baseline_run RUN_DIR matches the encoder tree on disk, e.g. .../bert_large_fullft_... vs roberta_large_...)
#   encoder mech/cue OUTPUT_DIR: data/<DATASET>/[/<out_variant>/]<VALIDATION_ROOT_ENCODER_BN>_encoder/…
#     (e.g. …/DynaHate/roberta_large_encoder/encoder_fft_cue_span_*). BERT vs RoBERTa never share the same run folder.
#   ENCODER_FFT_OUT_FLAT=1  (optional: write encoder_fft_* directly under data/<DATASET>/ like older submits — **collision
#     risk** between bert_large and roberta_large; default 0 = use …/<encoder_bn>_encoder/ subdir)
#   HPLOGIN_DGX_NODE_CSV=hpc-dgx-b-2,hpc-dgx-b-3  (hpclogin_dgx only; default pool; hpc-dgx-b-1 stripped unless HPC_ALLOW_DGX_B1=1)
#   HPLOGIN_DGX_ROUND_ROBIN=1  (hpclogin_dgx: pin one host per sbatch; default 0 = Slurm picks from pool)
#   HPLOGIN_DGX_PROBE_CUDA_NODES=1  (hpclogin_dgx: drop pool nodes that fail nvidia-smi ssh probe before sbatch)
#   SBATCH_ALLOW_DGX_B3=1  (hpclogin_dgx: keep hpc-dgx-b-3 in the pool if you listed it)
#   HPC_USER, HPC_HOST, HPC_RA, ...
set -euo pipefail

# LLM context modes: nocontext | traincontext (legacy aliases norag/trainrag still accepted).
LLM_SEL_HARD_PREFIX="${LLM_SEL_HARD_PREFIX:-llm_selected_hard}"
# Layout: suffix_phi (default llm_selected_hard_phi_nocontext) | phi_first (llm_phi_selected_hard_nocontext).
LLM_SEL_HARD_LAYOUT="${LLM_SEL_HARD_LAYOUT:-suffix_phi}"
# Middle token in llm_<fam>_<MID>_nocontext<tag> (default selected_hard; GT adapt → selected_hard_prompt_GT).
LLM_SEL_HARD_MID="${LLM_SEL_HARD_MID:-selected_hard}"

_normalize_teacher_mode() {
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]' | xargs)" in
    norag|nocontext) echo "nocontext" ;;
    trainrag|traincontext|context_train|ragtrain) echo "traincontext" ;;
    both) echo "both" ;;
    *) echo "$1" ;;
  esac
}

_normalize_teacher_modes_csv() {
  local raw="$1" part norm
  local -a out=()
  raw="$(_normalize_teacher_mode "${raw}")"
  if [[ "${raw}" == "both" ]]; then
    echo "nocontext,traincontext"
    return 0
  fi
  IFS=',' read -r -a parts <<< "$1"
  for part in "${parts[@]}"; do
    part="$(echo "${part}" | xargs)"
    [[ -z "${part}" ]] && continue
    norm="$(_normalize_teacher_mode "${part}")"
    if [[ "${norm}" == "both" ]]; then
      echo "nocontext,traincontext"
      return 0
    fi
    out+=("${norm}")
  done
  if [[ ${#out[@]} -eq 0 ]]; then
    echo "nocontext,traincontext"
  else
    local IFS=,
    echo "${out[*]}"
  fi
}

_submit_mech_cue_usage() {
  cat <<'EOF'
Usage:
  bash submit_roberta_fft_mech_cue_post_llm_from_local.sh [seed] [dataset] [options]

Options (override env when passed on the command line):
  --seed, -s SEED              ECHOHATE_DATA_SEED_DIR (e.g. seed_0)
  --dataset, -d DATASET        ONLY_DATASETS (comma-separated ok)
  --approach APPROACH          mech only
  --teacher-modes MODES        nocontext | traincontext | both  (default: both)
  --cue-lambda-mech-grid GRID  CUE_LAMBDA_MECH_GRID (e.g. 0 for attn-alone / no mech head)
  --cue-only                   shorthand for --approach cue --cue-lambda-mech-grid 0 (attn alone)
  --mech-only                  shorthand for --approach mech (mech alone, no cue-span)
  --baseline-run-rel PATH      encoder baseline dir under data/ (one --dataset only)
  --encoder-run-tag TAG        ENCODER_RUN_TAG suffix on output dirs
  --only-datasets CSV          alias for --dataset
  --skip-if-output-exists      skip grid cells whose OUTPUT_DIR already has checkpoints on HPC
  -h, --help

Example (attn alone, one dataset):
  bash submit_roberta_fft_mech_cue_post_llm_from_local.sh \
    --seed seed_0_model --dataset IHC --cue-only --encoder-run-tag _attn_only

Example (mech alone):
  bash submit_roberta_fft_mech_cue_post_llm_from_local.sh \
    --seed seed_0_model --dataset IHC --mech-only --encoder-run-tag _mech_only

Example (warm-start from baseline best-epoch weights; dirs get *_warmft suffix):
  ENCODER_RUN_TAG=_warmft \\
  bash submit_roberta_fft_mech_cue_post_llm_from_local.sh \\
    --seed seed_0_model_woCL --dataset Ethos --approach cue --teacher-modes nocontext \\
    --encoder-run-tag _warmft --skip-if-output-exists
  # Probes data/<seed>/<DS>/roberta_large_baseline/model/pytorch_model.pt on HPC;
  # loads backbone + classifier into cue (separate dirs from from-scratch).

Example (legacy shorthand):
  bash submit_roberta_fft_mech_cue_post_llm_from_local.sh \
    --seed seed_0 --dataset IHC --approach cue --teacher-modes nocontext --cue-only
EOF
}

_submit_mech_cue_parse_cli() {
  local _pos_seed="" _pos_dataset="" _explicit_baseline=""
  _SUBMIT_CLI_SKIP_IF_OUTPUT_EXISTS=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        _submit_mech_cue_usage
        exit 0
        ;;
      --seed|-s)
        [[ $# -lt 2 ]] && { echo "ERROR: $1 requires a value" >&2; exit 1; }
        ECHOHATE_DATA_SEED_DIR="$2"
        shift 2
        ;;
      --dataset|-d|--only-datasets)
        [[ $# -lt 2 ]] && { echo "ERROR: $1 requires a value" >&2; exit 1; }
        ONLY_DATASETS="$2"
        shift 2
        ;;
      --approach)
        [[ $# -lt 2 ]] && { echo "ERROR: $1 requires a value" >&2; exit 1; }
        APPROACH="$2"
        shift 2
        ;;
      --teacher-modes)
        [[ $# -lt 2 ]] && { echo "ERROR: $1 requires a value" >&2; exit 1; }
        TEACHER_MODES="$2"
        shift 2
        ;;
      --cue-lambda-mech-grid)
        [[ $# -lt 2 ]] && { echo "ERROR: $1 requires a value" >&2; exit 1; }
        CUE_LAMBDA_MECH_GRID="$2"
        shift 2
        ;;
      --cue-only)
        echo "ERROR: --cue-only / APPROACH=cue not supported; use APPROACH=mech." >&2
        echo "  Use --mech-only / APPROACH=mech." >&2
        exit 1
        ;;
      --mech-only)
        APPROACH=mech
        shift
        ;;
      --baseline-run-rel|--encoder-baseline-run-rel)
        [[ $# -lt 2 ]] && { echo "ERROR: $1 requires a value" >&2; exit 1; }
        _explicit_baseline="$2"
        shift 2
        ;;
      --encoder-run-tag)
        [[ $# -lt 2 ]] && { echo "ERROR: $1 requires a value" >&2; exit 1; }
        ENCODER_RUN_TAG="$2"
        shift 2
        ;;
      --skip-if-output-exists)
        _SUBMIT_CLI_SKIP_IF_OUTPUT_EXISTS=1
        shift
        ;;
      --)
        shift
        break
        ;;
      -*)
        echo "ERROR: unknown option: $1" >&2
        _submit_mech_cue_usage >&2
        exit 1
        ;;
      *)
        if [[ -z "${_pos_seed}" ]] && [[ "$1" == seed_* ]]; then
          _pos_seed="$1"
          ECHOHATE_DATA_SEED_DIR="$1"
        elif [[ -z "${_pos_dataset}" ]]; then
          _pos_dataset="$1"
          ONLY_DATASETS="$1"
        else
          echo "ERROR: unexpected positional argument: $1" >&2
          _submit_mech_cue_usage >&2
          exit 1
        fi
        shift
        ;;
    esac
  done
  SUBMIT_CLI_EXPLICIT_BASELINE="${_explicit_baseline}"
}

_submit_mech_cue_apply_baseline_rel_for_dataset() {
  local _ds="$1"
  local _rel="$2"
  case "${_ds}" in
    IHC) ENCODER_IHC_BASELINE_RUN_REL="${_rel}" ;;
    DynaHate) ENCODER_DYNAHATE_BASELINE_RUN_REL="${_rel}" ;;
    HatEval) ENCODER_HATEVAL_BASELINE_RUN_REL="${_rel}" ;;
    HateXplain) ENCODER_HATEXPLAIN_BASELINE_RUN_REL="${_rel}" ;;
    Ethos) ENCODER_ETHOS_BASELINE_RUN_REL="${_rel}" ;;
    offensive_as_hate) ENCODER_OFFENSIVE_AS_HATE_BASELINE_RUN_REL="${_rel}" ;;
    offensive_as_not_hate) ENCODER_OFFENSIVE_AS_NOT_HATE_BASELINE_RUN_REL="${_rel}" ;;
    offensive_as_not_hate_soft_none) ENCODER_OFFENSIVE_AS_NOT_HATE_SOFT_NONE_BASELINE_RUN_REL="${_rel}" ;;
    drop_offensive) ENCODER_DROP_OFFENSIVE_BASELINE_RUN_REL="${_rel}" ;;
    drop_offensive_repo_split) ENCODER_DROP_OFFENSIVE_REPO_SPLIT_BASELINE_RUN_REL="${_rel}" ;;
    SBIC) ENCODER_SBIC_BASELINE_RUN_REL="${_rel}" ;;
    ToxiGen|toxigen_new) ENCODER_TOXIGEN_BASELINE_RUN_REL="${_rel}" ;;
    white_new) ENCODER_WHITE_NEW_BASELINE_RUN_REL="${_rel}" ;;
    *)
      echo "[submit_roberta_fft_mech_cue] WARN: no ENCODER_*_BASELINE_RUN_REL slot for dataset=${_ds}; use --baseline-run-rel with a known name" >&2
      ;;
  esac
}

_random_state_from_seed_dir() {
  local _sd="$1"
  # seed_0_model, seed_0_model_wCL, seed_0_model_woCL, seed_0_model_new, …
  if [[ "${_sd}" =~ ^seed_([0-9]+)_model ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "${_sd}" =~ ^seed_([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  echo "0"
}

_echohate_align_random_state_to_seed_bucket() {
  if [[ "${RANDOM_STATE_OVERRIDE:-0}" == "1" ]]; then
    return 0
  fi
  if [[ -z "${ECHOHATE_DATA_SEED_DIR:-}" ]] || [[ ! "${ECHOHATE_DATA_SEED_DIR}" =~ ^seed_[0-9]+(_model.*)?$ ]]; then
    return 0
  fi
  local _rs
  _rs="$(_random_state_from_seed_dir "${ECHOHATE_DATA_SEED_DIR}")"
  if [[ "${RANDOM_STATE:-}" != "${_rs}" ]]; then
    echo "[submit_roberta_fft_mech_cue] RANDOM_STATE=${RANDOM_STATE:-<unset>} -> ${_rs} (from ${ECHOHATE_DATA_SEED_DIR})" >&2
    RANDOM_STATE="${_rs}"
    SEED="${_rs}"
  fi
}

_submit_mech_cue_autofill_baseline_rels() {
  local _seed="${ECHOHATE_DATA_SEED_DIR:-seed_0_model}"
  local _rb="${ECHOHATE_ROBERTA_BASELINE_DIR:-roberta_large_baseline}"
  local _ds_csv="${ONLY_DATASETS:-}"
  if [[ -n "${SUBMIT_CLI_EXPLICIT_BASELINE:-}" ]] && [[ -z "${_ds_csv}" ]]; then
    echo "ERROR: --baseline-run-rel requires --dataset" >&2
    exit 1
  fi
  [[ -z "${_ds_csv}" ]] && return 0
  local _rel _ds _existing
  IFS=',' read -r -a _ds_arr <<< "${_ds_csv}"
  for _ds in "${_ds_arr[@]}"; do
    _ds="$(echo "${_ds}" | xargs)"
    [[ -z "${_ds}" ]] && continue
    # Keep caller-exported ENCODER_*_BASELINE_RUN_REL (e.g. eacl …/bert_large_uncased_baseline_woCL).
    # Only fill empty slots; --baseline-run-rel always wins.
    if [[ -z "${SUBMIT_CLI_EXPLICIT_BASELINE:-}" ]]; then
      _existing=""
      case "${_ds}" in
        IHC) _existing="${ENCODER_IHC_BASELINE_RUN_REL:-}" ;;
        DynaHate) _existing="${ENCODER_DYNAHATE_BASELINE_RUN_REL:-}" ;;
        HatEval) _existing="${ENCODER_HATEVAL_BASELINE_RUN_REL:-}" ;;
        HateXplain) _existing="${ENCODER_HATEXPLAIN_BASELINE_RUN_REL:-}" ;;
        Ethos) _existing="${ENCODER_ETHOS_BASELINE_RUN_REL:-}" ;;
        offensive_as_hate) _existing="${ENCODER_OFFENSIVE_AS_HATE_BASELINE_RUN_REL:-}" ;;
        offensive_as_not_hate) _existing="${ENCODER_OFFENSIVE_AS_NOT_HATE_BASELINE_RUN_REL:-}" ;;
        offensive_as_not_hate_soft_none) _existing="${ENCODER_OFFENSIVE_AS_NOT_HATE_SOFT_NONE_BASELINE_RUN_REL:-}" ;;
        drop_offensive) _existing="${ENCODER_DROP_OFFENSIVE_BASELINE_RUN_REL:-}" ;;
        drop_offensive_repo_split) _existing="${ENCODER_DROP_OFFENSIVE_REPO_SPLIT_BASELINE_RUN_REL:-}" ;;
        SBIC) _existing="${ENCODER_SBIC_BASELINE_RUN_REL:-}" ;;
        ToxiGen|toxigen_new) _existing="${ENCODER_TOXIGEN_BASELINE_RUN_REL:-}" ;;
        white_new) _existing="${ENCODER_WHITE_NEW_BASELINE_RUN_REL:-}" ;;
      esac
      if [[ -n "${_existing}" ]]; then
        continue
      fi
    fi
    _rel="data/${_seed}/${_ds}/${_rb}"
    if [[ -n "${SUBMIT_CLI_EXPLICIT_BASELINE:-}" ]]; then
      if [[ "${#_ds_arr[@]}" -ne 1 ]]; then
        echo "ERROR: --baseline-run-rel requires exactly one --dataset (got ${ONLY_DATASETS})" >&2
        exit 1
      fi
      _rel="${SUBMIT_CLI_EXPLICIT_BASELINE}"
    fi
    _submit_mech_cue_apply_baseline_rel_for_dataset "${_ds}" "${_rel}"
  done
}

_submit_mech_cue_parse_cli "$@"

_BOOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_BOOT_DIR}/../../hpc_defaults_from_local.sh"
# shellcheck source=/dev/null
source "${_BOOT_DIR}/../../echohate_project_env.sh"
# shellcheck source=/dev/null
source "${_BOOT_DIR}/../../training_hyper_defaults.sh"
hpc_apply_ra_default_or_resolve
ECHOHATE_ROBERTA_BASELINE_DIR="${ECHOHATE_ROBERTA_BASELINE_DIR:-roberta_large_baseline}"
_submit_mech_cue_autofill_baseline_rels

HPC_DIR="${HPC_RA}/experiment/echohate"
# Passed to Slurm as DATA_ROOT (train/validation CSVs = ${DATA_ROOT}/${DATASET}/${ECHOHATE_SPLITTED_DATA_SUBDIR}/).
# Default ${HPC_DIR}/data. When a baseline_run override starts with data/seed_<n>/..., auto-use ${HPC_DIR}/data/seed_<n>.
ENCODER_DATA_ROOT="${ENCODER_DATA_ROOT:-${HPC_DIR}/data}"
# When ECHOHATE_DATA_SEED_DIR is set (seed_0_model, eacl_seed0, …), Slurm CSVs live under data/<bucket>/.
if [[ -n "${ECHOHATE_DATA_SEED_DIR:-}" ]]; then
  ENCODER_DATA_ROOT="${HPC_DIR}/data/${ECHOHATE_DATA_SEED_DIR}"
fi
_encoder_data_root_from_baseline_rel() {
  local _rel="${1#./}"
  # data/<bucket>/<DS>/… → DATA_ROOT=…/data/<bucket> (seed_* or eacl_seed0, etc.)
  if [[ "${_rel}" == data/*/* ]]; then
    local _pre="${_rel#data/}"
    local _seed_bucket="${_pre%%/*}"
    ENCODER_DATA_ROOT="${HPC_DIR}/data/${_seed_bucket}"
  fi
}
[[ -n "${ENCODER_IHC_BASELINE_RUN_REL:-}" ]] && _encoder_data_root_from_baseline_rel "${ENCODER_IHC_BASELINE_RUN_REL}"
[[ -n "${ENCODER_DYNAHATE_BASELINE_RUN_REL:-}" ]] && _encoder_data_root_from_baseline_rel "${ENCODER_DYNAHATE_BASELINE_RUN_REL}"
[[ -n "${ENCODER_HATEVAL_BASELINE_RUN_REL:-}" ]] && _encoder_data_root_from_baseline_rel "${ENCODER_HATEVAL_BASELINE_RUN_REL}"
[[ -n "${ENCODER_HATEXPLAIN_BASELINE_RUN_REL:-}" ]] && _encoder_data_root_from_baseline_rel "${ENCODER_HATEXPLAIN_BASELINE_RUN_REL}"
[[ -n "${ENCODER_ETHOS_BASELINE_RUN_REL:-}" ]] && _encoder_data_root_from_baseline_rel "${ENCODER_ETHOS_BASELINE_RUN_REL}"
[[ -n "${ENCODER_OFFENSIVE_AS_HATE_BASELINE_RUN_REL:-}" ]] && _encoder_data_root_from_baseline_rel "${ENCODER_OFFENSIVE_AS_HATE_BASELINE_RUN_REL}"
[[ -n "${ENCODER_DROP_OFFENSIVE_BASELINE_RUN_REL:-}" ]] && _encoder_data_root_from_baseline_rel "${ENCODER_DROP_OFFENSIVE_BASELINE_RUN_REL}"
[[ -n "${ENCODER_DROP_OFFENSIVE_REPO_SPLIT_BASELINE_RUN_REL:-}" ]] && _encoder_data_root_from_baseline_rel "${ENCODER_DROP_OFFENSIVE_REPO_SPLIT_BASELINE_RUN_REL}"
[[ -n "${ENCODER_SBIC_BASELINE_RUN_REL:-}" ]] && _encoder_data_root_from_baseline_rel "${ENCODER_SBIC_BASELINE_RUN_REL}"
[[ -n "${ENCODER_TOXIGEN_BASELINE_RUN_REL:-}" ]] && _encoder_data_root_from_baseline_rel "${ENCODER_TOXIGEN_BASELINE_RUN_REL}"
[[ -n "${ENCODER_WHITE_NEW_BASELINE_RUN_REL:-}" ]] && _encoder_data_root_from_baseline_rel "${ENCODER_WHITE_NEW_BASELINE_RUN_REL}"
CONDA_ENV="${CONDA_ENV:-rag_v8_gpu}"

# Do not inherit MODEL_NAME=microsoft/phi-4 (or similar) from Phi llm submit shells.
_DEFAULT_STUDENT_ENCODER="roberta-large"
if [[ -n "${STUDENT_MODEL_NAME:-}" ]]; then
  MODEL_NAME="${STUDENT_MODEL_NAME}"
else
  _mn="${MODEL_NAME:-}"
  if [[ -z "${_mn}" ]]; then
    MODEL_NAME="${_DEFAULT_STUDENT_ENCODER}"
  else
    _ml="$(echo "${_mn}" | tr '[:upper:]' '[:lower:]')"
    case "${_ml}" in
      microsoft/phi*|*/phi-2|*/phi-3|*/phi-4|*/phi3|*/phi4|meta-llama/*|mistralai/*|google/gemma*|*qwen*)
        echo "[submit_roberta_fft_mech_cue] WARNING: MODEL_NAME=${_mn} looks like a llm LLM; using ${_DEFAULT_STUDENT_ENCODER} for encoder mech/cue FFT." >&2
        MODEL_NAME="${_DEFAULT_STUDENT_ENCODER}"
        ;;
      *)
        MODEL_NAME="${_mn}"
        ;;
    esac
  fi
fi

# On-disk baseline_run roots use bert_large / roberta_large style basenames (see llm submit matrices).
_mn_lc="$(echo "${MODEL_NAME:-}" | tr '[:upper:]' '[:lower:]')"
case "${_mn_lc}" in
  bert-large*|bert_large*)
    _VALIDATION_ENC_DEFAULT="bert_large"
    ;;
  bert-base*|bert_base*)
    _VALIDATION_ENC_DEFAULT="bert_base"
    ;;
  roberta-large*|roberta_large*)
    _VALIDATION_ENC_DEFAULT="roberta_large"
    ;;
  roberta-base*|roberta_base*)
    _VALIDATION_ENC_DEFAULT="roberta_base"
    ;;
  *)
    _VALIDATION_ENC_DEFAULT="roberta_large"
    ;;
esac
VALIDATION_ROOT_ENCODER_BN="${VALIDATION_ROOT_ENCODER_BN:-${_VALIDATION_ENC_DEFAULT}}"
_E="${VALIDATION_ROOT_ENCODER_BN}"
ENCODER_FFT_OUT_FLAT="${ENCODER_FFT_OUT_FLAT:-0}"

if [[ "${ECHOHATE_HPC_PROFILE:-}" == "hpclogin_dgx" ]]; then
  SBATCH_NODES="${SBATCH_NODES:-${HPLOGIN_DGX_NODE_CSV:-hpc-dgx-b-2,hpc-dgx-b-3}}"
  SBATCH_NODES="$(echohate_hpclogin_dgx_sanitize_nodes "${SBATCH_NODES}")"
else
  SBATCH_NODES="${SBATCH_NODES:-g46-1gpu-4,g46-1gpu-3}"
fi
APPROACH="${APPROACH:-mech}"
# CLI may have set TEACHER_MODES / CUE_LAMBDA_MECH_GRID / ONLY_DATASETS before defaults below.
case "${APPROACH}" in
  mech) ;;
  cue|both)
    echo "[submit_roberta_fft_mech_cue] ERROR: APPROACH=${APPROACH} not supported; use APPROACH=mech." >&2
    echo "  Use APPROACH=mech for mechanism-distill jobs only." >&2
    exit 1
    ;;
  *)
    echo "[submit_roberta_fft_mech_cue] ERROR: APPROACH must be mech (got ${APPROACH})." >&2
    exit 1
    ;;
esac
TEACHER_MODES="$(_normalize_teacher_modes_csv "${TEACHER_MODES:-both}")"
TEACHER_MODES_ORDER="${TEACHER_MODES_ORDER:-interleaved}"
case "${TEACHER_MODES_ORDER}" in
  norag_first) TEACHER_MODES_ORDER="nocontext_first" ;;
esac
ONLY_DATASETS="${ONLY_DATASETS:-}"
ONLY_ETHOS_VARIANT="${ONLY_ETHOS_VARIANT:-both}"
GRID_PRESET="${GRID_PRESET:-}"

# Full grid (default): λ_mech = 0.01,0.05,0.1,0.5.
# Tiny grid (opt-in GRID_PRESET=tiny): denser mech-only λ for small-n pilots.
_DEFAULT_GRID="0.01,0.05,0.1,0.5"
_DEFAULT_LAMBDA_FULL="${_DEFAULT_GRID}"
_DEFAULT_LAMBDA_TINY="0.01,0.02,0.05,0.09,0.1,0.15"
_DEFAULT_EINIT_FULL="${_DEFAULT_GRID}"
_DEFAULT_EINIT_TINY="0.5,0.75,1.0,1.25"
_DEFAULT_CUE_LAMBDA_MECH_GRID_FULL="${_DEFAULT_GRID}"

_use_tiny=0
if [[ "${GRID_PRESET}" == "tiny" ]]; then
  _use_tiny=1
elif [[ "${GRID_PRESET}" == "full" ]]; then
  _use_tiny=0
fi

if [[ "${_use_tiny}" == "1" ]]; then
  LAMBDA_MECH_GRID="${LAMBDA_MECH_GRID:-${_DEFAULT_LAMBDA_TINY}}"
  E_INIT_GRID="${E_INIT_GRID:-${_DEFAULT_EINIT_TINY}}"
else
  LAMBDA_MECH_GRID="${LAMBDA_MECH_GRID:-${_DEFAULT_LAMBDA_FULL}}"
  E_INIT_GRID="${E_INIT_GRID:-${_DEFAULT_EINIT_FULL}}"
fi
# Optional closed target_group aux (per-dataset N). Default off (single 0).
# Paper 11-grid: 0.01,0.03,0.05,0.07,0.09,0.1,0.3,0.5,0.7,0.9,1.0
LAMBDA_TARGET_GRID="${LAMBDA_TARGET_GRID:-0}"
# Target-aux removed from mech_distill; only λ_target=0 is valid.
if [[ "${LAMBDA_TARGET_GRID}" != "0" && "${LAMBDA_TARGET_GRID}" != "0.0" ]]; then
  echo "WARN: LAMBDA_TARGET_GRID=${LAMBDA_TARGET_GRID} ignored; forcing 0" >&2
  LAMBDA_TARGET_GRID=0
fi
TARGET_NONE_POLICY="${TARGET_NONE_POLICY:-mask}"
LAMBDA_TARGET_RAMP_RATIO="${LAMBDA_TARGET_RAMP_RATIO:-0.1}"

# Cue scale: shared (one e) vs split (a=attack, t=target). Split uses A_INIT_GRID × T_INIT_GRID.
E_SCALE_MODE="$(echo "${E_SCALE_MODE:-shared}" | tr '[:upper:]' '[:lower:]')"
case "${E_SCALE_MODE}" in
  shared|union|single|e) E_SCALE_MODE="shared" ;;
  split|separate|at|a_t|target_attack) E_SCALE_MODE="split" ;;
  *)
    echo "ERROR: E_SCALE_MODE must be shared|split (got ${E_SCALE_MODE})" >&2
    exit 1
    ;;
esac
_DEFAULT_AT_INIT="${_DEFAULT_GRID}"
if [[ "${E_SCALE_MODE}" == "split" ]]; then
  A_INIT_GRID="${A_INIT_GRID:-${_DEFAULT_AT_INIT}}"
  T_INIT_GRID="${T_INIT_GRID:-${_DEFAULT_AT_INIT}}"
fi

# CE↔contrastive mix from seed-bucket name.
#   seed_*_woCL → CE only (export sentinel "none" — empty --export= can be dropped by Slurm)
#   seed_*_wCL  → λ=0.5
# For these ablation buckets the folder ALWAYS wins (ambient shell λ ignored) unless
# CE_CONTRASTIVE_LAMBDA_EXPLICIT=1. Match *_woCL before *_wCL (woCL contains the substring wCL).
_ce_folder_expected=""
_ce_folder_kind=""
case "${ECHOHATE_DATA_SEED_DIR:-}" in
  *_woCL)
    _ce_folder_expected=""
    _ce_folder_kind="woCL"
    ;;
  *_wCL)
    _ce_folder_expected="0.5"
    _ce_folder_kind="wCL"
    ;;
esac
if [[ -n "${_ce_folder_kind}" ]]; then
  if [[ "${CE_CONTRASTIVE_LAMBDA_EXPLICIT:-0}" == "1" ]]; then
    echo "[submit_roberta_fft_mech_cue] CE_CONTRASTIVE_LAMBDA_EXPLICIT=1 — keeping CE_CONTRASTIVE_LAMBDA='${CE_CONTRASTIVE_LAMBDA:-}' (folder=${_ce_folder_kind})"
  else
    if [[ -n "${CE_CONTRASTIVE_LAMBDA+x}" ]] && [[ "${CE_CONTRASTIVE_LAMBDA}" != "${_ce_folder_expected}" ]] \
      && ! { [[ -z "${_ce_folder_expected}" ]] && [[ "${CE_CONTRASTIVE_LAMBDA}" == "none" || "${CE_CONTRASTIVE_LAMBDA}" == "off" ]]; }; then
      echo "[submit_roberta_fft_mech_cue] WARN: ambient CE_CONTRASTIVE_LAMBDA='${CE_CONTRASTIVE_LAMBDA}' ignored for ${_ce_folder_kind} → '${_ce_folder_expected:-(CE only)}'"
    fi
    CE_CONTRASTIVE_LAMBDA="${_ce_folder_expected}"
  fi
elif [[ -z "${CE_CONTRASTIVE_LAMBDA+x}" ]]; then
  CE_CONTRASTIVE_LAMBDA=
fi
# Slurm-safe: never export an empty CE_CONTRASTIVE_LAMBDA= (ALL can keep a stale value).
if [[ -z "${CE_CONTRASTIVE_LAMBDA}" ]]; then
  CE_CONTRASTIVE_LAMBDA=none
fi
# Hard fail if ablation folder and λ still disagree after resolution.
case "${ECHOHATE_DATA_SEED_DIR:-}" in
  *_woCL)
    if [[ "${CE_CONTRASTIVE_LAMBDA}" != "none" && "${CE_CONTRASTIVE_LAMBDA}" != "off" && -n "${CE_CONTRASTIVE_LAMBDA}" ]]; then
      echo "ERROR: seed ${ECHOHATE_DATA_SEED_DIR} is woCL (CE-only) but CE_CONTRASTIVE_LAMBDA='${CE_CONTRASTIVE_LAMBDA}'" >&2
      exit 1
    fi
    ;;
  *_wCL)
    if [[ -z "${CE_CONTRASTIVE_LAMBDA}" || "${CE_CONTRASTIVE_LAMBDA}" == "none" || "${CE_CONTRASTIVE_LAMBDA}" == "off" ]]; then
      echo "ERROR: seed ${ECHOHATE_DATA_SEED_DIR} is wCL but CE_CONTRASTIVE_LAMBDA is empty/none" >&2
      exit 1
    fi
    ;;
esac
echo "[submit_roberta_fft_mech_cue] CE_CONTRASTIVE_LAMBDA='${CE_CONTRASTIVE_LAMBDA}' (folder=${ECHOHATE_DATA_SEED_DIR:-<unset>} kind=${_ce_folder_kind:-other})"
CONTRASTIVE_MARGIN="${CONTRASTIVE_MARGIN:-0.75}"
# Cue mechanism λ grid: default 4 values (× default E_INIT = 4 → 16 cue sbatch cells per dataset×mode).
# Override CUE_LAMBDA_MECH only (no GRID) → single λ column (legacy). Set CUE_LAMBDA_MECH_GRID explicitly for a custom list.
if [[ -n "${CUE_LAMBDA_MECH_GRID:-}" ]]; then
  CUE_LAMBDA_MECH="${CUE_LAMBDA_MECH:-0.0}"
elif [[ -n "${CUE_LAMBDA_MECH:-}" ]]; then
  CUE_LAMBDA_MECH_GRID="${CUE_LAMBDA_MECH}"
else
  CUE_LAMBDA_MECH_GRID="${_DEFAULT_CUE_LAMBDA_MECH_GRID_FULL}"
  CUE_LAMBDA_MECH="${CUE_LAMBDA_MECH:-0.01}"
fi
CUE_LAMBDA_MECH_RAMP_RATIO="${CUE_LAMBDA_MECH_RAMP_RATIO:-0.1}"
CUE_MODE="${CUE_MODE:-attention_mask_mode}"
RANDOM_STATE="${RANDOM_STATE:-0}"
_echohate_align_random_state_to_seed_bucket
# Default off (always submit). Opt-in only: --skip-if-output-exists (ignores stale export in parent shell).
SKIP_IF_OUTPUT_EXISTS=0
if [[ "${_SUBMIT_CLI_SKIP_IF_OUTPUT_EXISTS:-0}" == "1" ]]; then
  SKIP_IF_OUTPUT_EXISTS=1
fi
RUN_EVAL_OVERWRITE_OUTPUTS="${RUN_EVAL_OVERWRITE_OUTPUTS:-1}"
TRACK_DYNAMICS="${TRACK_DYNAMICS:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
TEACHER_OUTPUT_TAG="${TEACHER_OUTPUT_TAG:-}"
TEACHER_OUTPUT_TAG_OVERRIDES="${TEACHER_OUTPUT_TAG_OVERRIDES:-}"
TEACHER_LM_FAMILY="${TEACHER_LM_FAMILY:-}"
BERT_PHI_NEW_PROMPT_TAG="${BERT_PHI_NEW_PROMPT_TAG:-_new_prompt}"
BERT_PHI_NEW_PROMPT_ONLY_DATASETS="${BERT_PHI_NEW_PROMPT_ONLY_DATASETS:-IHC,DynaHate,HatEval}"
ENCODER_RUN_TAG="${ENCODER_RUN_TAG:-}"
INIT_BACKBONE_AUTO="${INIT_BACKBONE_AUTO:-0}"
INIT_LOAD_CLASSIFIER="${INIT_LOAD_CLASSIFIER:-1}"
export HATE_MECH_TAXONOMY=8way
echo "[submit_roberta_fft_mech_cue] HATE_MECH_TAXONOMY=${HATE_MECH_TAXONOMY}"
echo "[submit_roberta_fft_mech_cue] MECH_AUX_MODE=${MECH_AUX_MODE:-mechanism}"
# Amplify channel filter (attack-only / target-only). Keep dirs distinct from union attn-only.
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
if [[ "${SPAN_CHANNELS}" == "attack" ]]; then
  if [[ -z "${ENCODER_RUN_TAG}" || "${ENCODER_RUN_TAG}" == "_attn_only" ]]; then
    ENCODER_RUN_TAG="_attn_only_attack"
    echo "[submit_roberta_fft_mech_cue] SPAN_CHANNELS=attack → ENCODER_RUN_TAG=${ENCODER_RUN_TAG}" >&2
  elif [[ "${ENCODER_RUN_TAG}" != *attack* ]]; then
    ENCODER_RUN_TAG="${ENCODER_RUN_TAG}_attack"
    echo "[submit_roberta_fft_mech_cue] SPAN_CHANNELS=attack → ENCODER_RUN_TAG=${ENCODER_RUN_TAG}" >&2
  fi
elif [[ "${SPAN_CHANNELS}" == "target" ]]; then
  if [[ -z "${ENCODER_RUN_TAG}" || "${ENCODER_RUN_TAG}" == "_attn_only" ]]; then
    ENCODER_RUN_TAG="_attn_only_target"
    echo "[submit_roberta_fft_mech_cue] SPAN_CHANNELS=target → ENCODER_RUN_TAG=${ENCODER_RUN_TAG}" >&2
  elif [[ "${ENCODER_RUN_TAG}" != *target* ]]; then
    ENCODER_RUN_TAG="${ENCODER_RUN_TAG}_target"
    echo "[submit_roberta_fft_mech_cue] SPAN_CHANNELS=target → ENCODER_RUN_TAG=${ENCODER_RUN_TAG}" >&2
  fi
fi
if [[ -n "${INIT_BACKBONE_FROM:-}" ]] || [[ "${INIT_BACKBONE_AUTO:-0}" == "1" ]] || [[ "${ENCODER_RUN_TAG}" == *warmft* ]]; then
  echo "[submit_roberta_fft_mech_cue] ERROR: warmft / INIT_BACKBONE_FROM not supported; train from scratch." >&2
  exit 1
fi
_WARMFT_MODE=0
# Per-row: CURRENT_LLM_OUTPUT_TAG + ENCODER_RUN_TAG (set inside RUN_SPECS loop via resolve_current_llm_output_tag).
STUDENT_OUT_SUFFIX=""

# User-configured llm lookup (saved before optional per-profile overrides / bert+phi merge).
_BASE_TEACHER_OUTPUT_TAG="${TEACHER_OUTPUT_TAG:-}"
_BASE_TEACHER_OUTPUT_TAG_OVERRIDES="${TEACHER_OUTPUT_TAG_OVERRIDES:-}"
_USER_TEACHER_LM_FAMILY="${TEACHER_LM_FAMILY:-}"
SUBMIT_STUDENT_TEACHER_PROFILES="${SUBMIT_STUDENT_TEACHER_PROFILES:-}"
TEACHER_PATH_SLUG="phi"
if [[ "${CUE_MODE}" != "attention_mask_mode" ]]; then
  echo "[submit_roberta_fft_mech_cue] ERROR: only CUE_MODE=attention_mask_mode is supported (got ${CUE_MODE})." >&2
  exit 1
fi
CHECK_TEACHER_AT_SUBMIT="${CHECK_TEACHER_AT_SUBMIT:-0}"
CHECK_INPUTS_AT_SUBMIT="${CHECK_INPUTS_AT_SUBMIT:-1}"
CHECK_INPUTS_STRICT="${CHECK_INPUTS_STRICT:-0}"
AUTO_FIX_TRAINID_FROM_TEXT="${AUTO_FIX_TRAINID_FROM_TEXT:-1}"
# Cue Slurm only: passed to slurm_train_cue_span_single.sh (OOM mitigation).
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
PACK_LAMBDA_GRID="${PACK_LAMBDA_GRID:-0}"
case "${PACK_LAMBDA_GRID}" in
  1|true|TRUE|yes|YES|on|ON) PACK_LAMBDA_GRID=1 ;;
  *) PACK_LAMBDA_GRID=0 ;;
esac
if [[ "${ECHOHATE_HPC_PROFILE:-}" == "hpclogin_dgx" ]]; then
  SBATCH_PARTITION="${SBATCH_PARTITION:-dgx}"
  case "${SBATCH_PARTITION}" in
    gpu | dgx-1 | dgx-2) SBATCH_PARTITION=dgx ;;
  esac
else
  SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
fi
SBATCH_CPUS="${SBATCH_CPUS:-8}"
SBATCH_MEM="${SBATCH_MEM:-96G}"

SLURM_MECH="${SLURM_MECH:-job_submission/stage2/cue_mech_attnmask/slurm_mech_distill_single.sh}"
# Cue Slurm unused; keep unset.
unset SLURM_CUE 2>/dev/null || true

if [ -f "${_BOOT_DIR}/../../../pipeline/stage2/mech_distill.py" ] \
  || [ -f "${_BOOT_DIR}/../../../config/echohate_project_config.json" ]; then
  LOCAL_ECHOHATE="$(cd "${_BOOT_DIR}/../../.." && pwd)"
elif [ -d "experiment/echohate/pipeline" ] || [ -f "experiment/echohate/config/echohate_project_config.json" ]; then
  LOCAL_ECHOHATE="$(pwd)/experiment/echohate"
elif [ -d "pipeline/stage2" ] || [ -f "config/echohate_project_config.json" ]; then
  LOCAL_ECHOHATE="$(pwd)"
elif [ -d "echohate/pipeline" ]; then
  LOCAL_ECHOHATE="$(pwd)/echohate"
else
  echo "Run from RA repo root or experiment/echohate (could not find pipeline/ or config/)." >&2
  exit 1
fi

echo "=========================================="
echo "encoder FFT post llm (selected-hard; APPROACH=mech)"
echo "HPC: ${HPC_USER}@${HPC_HOST}:${HPC_DIR}"
echo "APPROACH: ${APPROACH}  TEACHER_MODES: ${TEACHER_MODES}  TEACHER_MODES_ORDER: ${TEACHER_MODES_ORDER}"
echo "MODEL_NAME: ${MODEL_NAME}  VALIDATION_ROOT_ENCODER_BN: ${_E}"
echo "ENCODER_FFT_OUT_FLAT: ${ENCODER_FFT_OUT_FLAT}  (0 → encoder outputs under …/data/<DS>/[variant/]/<encoder>_encoder/)"
echo "GRID_PRESET: ${GRID_PRESET:-auto}  (tiny=${_use_tiny})"
echo "LAMBDA_MECH_GRID: ${LAMBDA_MECH_GRID}"
echo "PACK_LAMBDA_GRID: ${PACK_LAMBDA_GRID}  (1 = one sbatch packs all λ per teacher×mode)"
echo "LAMBDA_TARGET_GRID: ${LAMBDA_TARGET_GRID}  TARGET_NONE_POLICY: ${TARGET_NONE_POLICY}  LAMBDA_TARGET_RAMP_RATIO: ${LAMBDA_TARGET_RAMP_RATIO}"
echo "E_INIT_GRID: ${E_INIT_GRID}"
echo "E_SCALE_MODE: ${E_SCALE_MODE}  A_INIT_GRID: ${A_INIT_GRID:-<n/a>}  T_INIT_GRID: ${T_INIT_GRID:-<n/a>}"
echo "CE_CONTRASTIVE_LAMBDA: ${CE_CONTRASTIVE_LAMBDA}  CONTRASTIVE_MARGIN: ${CONTRASTIVE_MARGIN}"
echo "CUE_LAMBDA_MECH: ${CUE_LAMBDA_MECH}  CUE_LAMBDA_MECH_RAMP_RATIO: ${CUE_LAMBDA_MECH_RAMP_RATIO}"
echo "CUE_LAMBDA_MECH_GRID: ${CUE_LAMBDA_MECH_GRID}"
echo "CUE_MODE: ${CUE_MODE}"
echo "Resolved defaults: epochs=${NUM_EPOCHS} batch=${BATCH_SIZE} eval_batch=${EVAL_BATCH_SIZE} lr=${LEARNING_RATE} wd=${WEIGHT_DECAY} warmup=${WARMUP_RATIO} max_len=${MAX_LENGTH} seed=${RANDOM_STATE}"
echo "ONLY_DATASETS: ${ONLY_DATASETS:-(all)}"
echo "ONLY_ETHOS_VARIANT: ${ONLY_ETHOS_VARIANT}"
echo "SBATCH_NODES: ${SBATCH_NODES}"
echo "SKIP_IF_OUTPUT_EXISTS: ${SKIP_IF_OUTPUT_EXISTS}"
echo "RUN_EVAL_OVERWRITE_OUTPUTS: ${RUN_EVAL_OVERWRITE_OUTPUTS}"
echo "FORCE_RETRAIN: ${FORCE_RETRAIN}"
echo "SUBMIT_STUDENT_TEACHER_PROFILES: ${SUBMIT_STUDENT_TEACHER_PROFILES:-<empty = single wave from MODEL_NAME / TEACHER_*>}"
echo "TEACHER_LM_FAMILY: ${TEACHER_LM_FAMILY:-<unset>}  BERT_PHI trim datasets: ${BERT_PHI_NEW_PROMPT_ONLY_DATASETS}"
echo "TEACHER_OUTPUT_TAG (fallback): ${TEACHER_OUTPUT_TAG:-<empty>}"
echo "TEACHER_OUTPUT_TAG_OVERRIDES: ${TEACHER_OUTPUT_TAG_OVERRIDES:-<none>}"
echo "ENCODER_DATA_ROOT (Slurm DATA_ROOT): ${ENCODER_DATA_ROOT}"
echo "ENCODER_RUN_TAG: ${ENCODER_RUN_TAG:-<none>}  (encoder dir suffix = resolved llm tag + ENCODER_RUN_TAG per dataset row)"
echo "SPAN_OP: ${SPAN_OP:-amplify}  SPAN_CHANNELS: ${SPAN_CHANNELS:-both}"
echo "WARMFT_MODE: ${_WARMFT_MODE}  INIT_BACKBONE_AUTO: ${INIT_BACKBONE_AUTO}  INIT_BACKBONE_FROM: ${INIT_BACKBONE_FROM:-<unset>}  INIT_LOAD_CLASSIFIER: ${INIT_LOAD_CLASSIFIER}"
echo "TRAIN_ON_TRAIN_VAL: ${TRAIN_ON_TRAIN_VAL:-0}  FIXED_THRESHOLD: ${FIXED_THRESHOLD:-<slurm default 0.5 if trainval>}  TUNE_SPLIT: ${TUNE_SPLIT:-<slurm default none if trainval>}"
echo "CHECK_TEACHER_AT_SUBMIT: ${CHECK_TEACHER_AT_SUBMIT}"
echo "CHECK_INPUTS_AT_SUBMIT: ${CHECK_INPUTS_AT_SUBMIT}  CHECK_INPUTS_STRICT: ${CHECK_INPUTS_STRICT}"
echo "GRADIENT_ACCUMULATION_STEPS (cue only): ${GRADIENT_ACCUMULATION_STEPS}  GRADIENT_CHECKPOINTING: ${GRADIENT_CHECKPOINTING}"
echo "=========================================="

hpc_mkdir_remote_from_local "${HPC_DIR}" "${HPC_DIR}/job_submission/mech_distill"

if [[ "${HPC_SKIP_CODE_RSYNC:-0}" == "1" ]]; then
  echo "[submit_roberta_fft_mech_cue] HPC_SKIP_CODE_RSYNC=1 — skipping code rsync."
else
  echo "Sync code (exclude data/)..."
  hpc_rsync_with_retry -avz --progress \
    --exclude="data/" \
    --exclude=".git/" \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    "${LOCAL_ECHOHATE}/" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/"
fi

llm_dir_for_mode() {
  local _t="${CURRENT_LLM_OUTPUT_TAG:-${TEACHER_OUTPUT_TAG:-}}"
  local _mode _layout="${LLM_SEL_HARD_LAYOUT}"
  local _fam="${TEACHER_LM_FAMILY:-phi}"
  local _mid="${LLM_SEL_HARD_MID:-selected_hard}"
  if [[ "${JOB_DS_TAG:-}" == "drop_offensive_repo_split" ]]; then
    _layout="phi_first"
    _fam="phi"
  fi
  _mode="$(_normalize_teacher_mode "$1")"
  # Legacy phi_first dirs (llm_phi_selected_hard_*). Non-phi teachers always use llm_<fam>_.
  if [[ "${_layout}" == "phi_first" ]] && [[ "${_fam}" == "phi" ]]; then
    case "${_mode}" in
      nocontext) echo "llm_phi_${_mid}_nocontext${_t}" ;;
      traincontext) echo "llm_phi_${_mid}_with_context_traincontext${_t}" ;;
      *)
        echo "ERROR: unknown TEACHER_MODE=$1 (use nocontext or traincontext)" >&2
        return 1
        ;;
    esac
    return 0
  fi
  # Matches adapt_toy_prompt_hard_for_mech / slurm_llm_infer_on_hard:
  #   llm_<fam>_<mid>_{nocontext|with_context_traincontext}<tag>
  case "${_mode}" in
    nocontext) echo "llm_${_fam}_${_mid}_nocontext${_t}" ;;
    traincontext) echo "llm_${_fam}_${_mid}_with_context_traincontext${_t}" ;;
    *)
      echo "ERROR: unknown TEACHER_MODE=$1 (use nocontext or traincontext)" >&2
      return 1
      ;;
  esac
}

# JOB_DS_TAG = DATASET or DATASET_outvariant (e.g. Ethos_posfreq). DATASET = first column of RUN_SPECS.
resolve_current_llm_output_tag() {
  local _jtag="$1"
  local _ds="$2"
  CURRENT_LLM_OUTPUT_TAG="${TEACHER_OUTPUT_TAG:-}"
  if [[ -z "${TEACHER_OUTPUT_TAG_OVERRIDES:-}" ]]; then
    STUDENT_OUT_SUFFIX="${CURRENT_LLM_OUTPUT_TAG}${ENCODER_RUN_TAG}"
    return 0
  fi
  IFS=',' read -r -a _tov_arr <<< "${TEACHER_OUTPUT_TAG_OVERRIDES}"
  for _pr in "${_tov_arr[@]}"; do
    _pr="$(echo "${_pr}" | xargs)"
    [[ -z "${_pr}" ]] && continue
    case "${_pr}" in *"="*) ;; *) continue ;; esac
    _key="${_pr%%=*}"
    _key="$(echo "${_key}" | xargs)"
    _val="${_pr#*=}"
    _val="$(echo "${_val}" | xargs)"
    if [[ "${_key}" == "${_jtag}" ]]; then
      CURRENT_LLM_OUTPUT_TAG="${_val}"
      STUDENT_OUT_SUFFIX="${CURRENT_LLM_OUTPUT_TAG}${ENCODER_RUN_TAG}"
      return 0
    fi
  done
  for _pr in "${_tov_arr[@]}"; do
    _pr="$(echo "${_pr}" | xargs)"
    [[ -z "${_pr}" ]] && continue
    case "${_pr}" in *"="*) ;; *) continue ;; esac
    _key="${_pr%%=*}"
    _key="$(echo "${_key}" | xargs)"
    _val="${_pr#*=}"
    _val="$(echo "${_val}" | xargs)"
    if [[ "${_key}" == "${_ds}" ]]; then
      CURRENT_LLM_OUTPUT_TAG="${_val}"
      STUDENT_OUT_SUFFIX="${CURRENT_LLM_OUTPUT_TAG}${ENCODER_RUN_TAG}"
      return 0
    fi
  done
  STUDENT_OUT_SUFFIX="${CURRENT_LLM_OUTPUT_TAG}${ENCODER_RUN_TAG}"
}

IFS=',' read -r -a NODES_RAW <<< "${SBATCH_NODES}"
NODES=()
for n in "${NODES_RAW[@]}"; do
  n="$(echo "${n}" | xargs)"
  [[ -n "${n}" ]] && NODES+=("${n}")
done
if [[ ${#NODES[@]} -lt 1 ]]; then
  echo "ERROR: SBATCH_NODES empty"
  exit 1
fi
NODE_IDX=0
pick_node() {
  PICKED_NODE="${NODES[$NODE_IDX]}"
  NODE_IDX=$(( (NODE_IDX + 1) % ${#NODES[@]} ))
}

# hpclogin_dgx: never land on hpc-dgx-b-1 (flaky CUDA) unless HPC_ALLOW_DGX_B1=1; strip b-3 unless SBATCH_ALLOW_DGX_B3=1.
if [[ "${ECHOHATE_HPC_PROFILE:-}" == "hpclogin_dgx" ]]; then
  _dgx_nodes_filtered=()
  for _n in "${NODES[@]}"; do
    if [[ "${HPC_ALLOW_DGX_B1:-0}" != "1" ]] && [[ "${KEEP_AMBIENT_NODES:-0}" != "1" ]] && [[ "${_n}" == "hpc-dgx-b-1" ]]; then
      continue
    fi
    if [[ "${SBATCH_ALLOW_DGX_B3:-0}" != "1" ]] && [[ "${_n}" == "hpc-dgx-b-3" ]]; then
      continue
    fi
    _dgx_nodes_filtered+=("${_n}")
  done
  NODES=("${_dgx_nodes_filtered[@]}")
fi
if [[ ${#NODES[@]} -lt 1 ]]; then
  echo "ERROR: SBATCH_NODES empty after parsing / hpclogin_dgx node filter" >&2
  exit 1
fi

# hpclogin DGX: optional preflight — exclude compute nodes that do not answer nvidia-smi (pool = rest).
_hpclogin_probe_node_cuda() {
  local _node="$1"
  _hpc_ssh "${HPC_USER}@${HPC_HOST}" \
    "ssh -o BatchMode=yes -o ConnectTimeout=20 '${_node}' 'command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1'" \
    </dev/null >/dev/null 2>&1
}

if [[ "${ECHOHATE_HPC_PROFILE:-}" == "hpclogin_dgx" ]] && [[ "${HPLOGIN_DGX_PROBE_CUDA_NODES:-1}" == "1" ]]; then
  _cuda_ok_nodes=()
  for _n in "${NODES[@]}"; do
    if _hpclogin_probe_node_cuda "${_n}"; then
      echo "[submit_roberta_fft_mech_cue] DGX probe OK (nvidia-smi): ${_n}"
      _cuda_ok_nodes+=("${_n}")
    else
      echo "[submit_roberta_fft_mech_cue] DGX probe SKIP (no nvidia-smi / unreachable): ${_n}" >&2
    fi
  done
  if [[ ${#_cuda_ok_nodes[@]} -ge 1 ]]; then
    NODES=("${_cuda_ok_nodes[@]}")
  else
    echo "[submit_roberta_fft_mech_cue] WARN: all DGX probes failed; keeping full pool ${NODES[*]} (Slurm may still place on allocated GPU)." >&2
  fi
fi

# Comma pool for Slurm: pass **--nodes=1** with a comma-separated --nodelist so Slurm allocates **one** host from the pool
# (without --nodes=1, two hostnames can be read as **2 nodes** → PartitionNodeLimit / PartitionConfig on dgx).
SBATCH_NODELIST_POOL=""
for _n in "${NODES[@]}"; do
  SBATCH_NODELIST_POOL="${SBATCH_NODELIST_POOL:+${SBATCH_NODELIST_POOL},}${_n}"
done
if [[ "${ECHOHATE_HPC_PROFILE:-}" == "hpclogin_dgx" ]]; then
  if [[ "${HPLOGIN_DGX_ROUND_ROBIN:-0}" == "1" ]]; then
    echo "[submit_roberta_fft_mech_cue] hpclogin_dgx round-robin across: ${NODES[*]} (one fixed node per sbatch)"
  else
    echo "[submit_roberta_fft_mech_cue] hpclogin_dgx pool --nodes=1 --nodelist=${SBATCH_NODELIST_POOL} (Slurm picks one available host)"
  fi
fi

_pick_sbatch_nodelist() {
  if [[ "${ECHOHATE_HPC_PROFILE:-}" == "hpclogin_dgx" ]] && [[ "${HPLOGIN_DGX_ROUND_ROBIN:-0}" != "1" ]]; then
    printf '%s' "${SBATCH_NODELIST_POOL}"
  else
    pick_node
    printf '%s' "${PICKED_NODE}"
  fi
}

_spec_ok() {
  local ds="$1"
  [[ -z "${ONLY_DATASETS}" ]] && return 0
  case ",${ONLY_DATASETS}," in *",${ds},"*) return 0 ;; *) return 1 ;; esac
}

_ethos_variant_ok() {
  local out_variant="$1"
  case "${ONLY_ETHOS_VARIANT}" in
    both) return 0 ;;
    posfreq) [[ "${out_variant}" == "posfreq" ]] && return 0 || return 1 ;;
    posnone) [[ "${out_variant}" == "posnone" ]] && return 0 || return 1 ;;
    *)
      echo "ERROR: ONLY_ETHOS_VARIANT must be both|posfreq|posnone, got '${ONLY_ETHOS_VARIANT}'" >&2
      exit 1
      ;;
  esac
}

_merge_bert_phi_tag_overrides() {
  local _tlf_lc
  _tlf_lc="$(echo "${TEACHER_LM_FAMILY:-}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${_E}" == "bert_large" ]] && [[ "${_tlf_lc}" == "phi" ]]; then
    local _bfdn _bfd_has _bfd_pr _bfd_key _bfd_ex
    IFS=',' read -r -a _bfd_arr <<< "${BERT_PHI_NEW_PROMPT_ONLY_DATASETS}"
    for _bfdn in "${_bfd_arr[@]}"; do
      _bfdn="$(echo "${_bfdn}" | xargs)"
      [[ -z "${_bfdn}" ]] && continue
      _bfd_has=0
      if [[ -n "${TEACHER_OUTPUT_TAG_OVERRIDES:-}" ]]; then
        IFS=',' read -r -a _bfd_ex <<< "${TEACHER_OUTPUT_TAG_OVERRIDES}"
        for _bfd_pr in "${_bfd_ex[@]}"; do
          _bfd_pr="$(echo "${_bfd_pr}" | xargs)"
          [[ -z "${_bfd_pr}" ]] || [[ "${_bfd_pr}" != *"="* ]] && continue
          _bfd_key="${_bfd_pr%%=*}"
          _bfd_key="$(echo "${_bfd_key}" | xargs)"
          if [[ "${_bfd_key}" == "${_bfdn}" ]]; then
            _bfd_has=1
            break
          fi
        done
      fi
      if [[ "${_bfd_has}" == "0" ]]; then
        TEACHER_OUTPUT_TAG_OVERRIDES="${TEACHER_OUTPUT_TAG_OVERRIDES:+${TEACHER_OUTPUT_TAG_OVERRIDES},}${_bfdn}=${BERT_PHI_NEW_PROMPT_TAG}"
      fi
    done
    echo "[submit_roberta_fft_mech_cue] bert_large + TEACHER_LM_FAMILY=phi: appended dirname suffix ${BERT_PHI_NEW_PROMPT_TAG} for ${BERT_PHI_NEW_PROMPT_ONLY_DATASETS} (see TEACHER_OUTPUT_TAG_OVERRIDES)" >&2
  fi
}

_set_encoder_encoder_from_bn_name() {
  local bn="$1"
  VALIDATION_ROOT_ENCODER_BN="${bn}"
  _E="${bn}"
  case "$(echo "${bn}" | tr '[:upper:]' '[:lower:]')" in
    bert_large) MODEL_NAME="bert-large-uncased" ;;
    roberta_large) MODEL_NAME="roberta-large" ;;
    bert_base) MODEL_NAME="bert-base-uncased" ;;
    roberta_base) MODEL_NAME="roberta-base" ;;
    *)
      echo "ERROR: unknown encoder VALIDATION_ROOT_ENCODER_BN='${bn}' in SUBMIT_STUDENT_TEACHER_PROFILES (use bert_large|roberta_large|bert_base|roberta_base)" >&2
      exit 1
      ;;
  esac
}

_apply_llm_vars_for_lm_family() {
  local fam="$1"
  local f_lc
  f_lc="$(echo "${fam}" | tr '[:upper:]' '[:lower:]')"
  case "${f_lc}" in
    phi|qwen|llama|luna) ;;
    *)
      echo "ERROR: llm LM family must be phi|qwen|llama|luna, got '${fam}'" >&2
      exit 1
      ;;
  esac
  TEACHER_LM_FAMILY="${f_lc}"
  TEACHER_PATH_SLUG="${f_lc}"
  TEACHER_OUTPUT_TAG="${_BASE_TEACHER_OUTPUT_TAG:-}"
  TEACHER_OUTPUT_TAG_OVERRIDES="${_BASE_TEACHER_OUTPUT_TAG_OVERRIDES:-}"
  if [[ "${TOY_PROMPT_HARD_TEACHER:-0}" != "1" ]]; then
    case "${f_lc}" in
      phi)
        [[ -n "${TEACHER_OUTPUT_TAG_PHI:-}" ]] && TEACHER_OUTPUT_TAG="${TEACHER_OUTPUT_TAG_PHI}"
        [[ -n "${TEACHER_OUTPUT_TAG_OVERRIDES_PHI:-}" ]] && TEACHER_OUTPUT_TAG_OVERRIDES="${TEACHER_OUTPUT_TAG_OVERRIDES_PHI}"
        ;;
      qwen)
        [[ -n "${TEACHER_OUTPUT_TAG_QWEN:-}" ]] && TEACHER_OUTPUT_TAG="${TEACHER_OUTPUT_TAG_QWEN}"
        [[ -n "${TEACHER_OUTPUT_TAG_OVERRIDES_QWEN:-}" ]] && TEACHER_OUTPUT_TAG_OVERRIDES="${TEACHER_OUTPUT_TAG_OVERRIDES_QWEN}"
        ;;
      llama)
        [[ -n "${TEACHER_OUTPUT_TAG_LLAMA:-}" ]] && TEACHER_OUTPUT_TAG="${TEACHER_OUTPUT_TAG_LLAMA}"
        [[ -n "${TEACHER_OUTPUT_TAG_OVERRIDES_LLAMA:-}" ]] && TEACHER_OUTPUT_TAG_OVERRIDES="${TEACHER_OUTPUT_TAG_OVERRIDES_LLAMA}"
        ;;
    esac
    _merge_bert_phi_tag_overrides
  fi
}

_refresh_run_specs_array() {
  local _seed="${ECHOHATE_DATA_SEED_DIR:-seed_0_model}"
  local _rb="${ECHOHATE_ROBERTA_BASELINE_DIR:-roberta_large_baseline}"
  local _run_rel="data/${_seed}"
  local _split="${ECHOHATE_SPLITTED_DATA_SUBDIR:-splitted_data_80_10_10}"
  local _bb _ds
  local -a _seed_all=(
    IHC DynaHate HatEval HateXplain Ethos offensive_as_hate offensive_as_not_hate
    offensive_as_not_hate_soft_none drop_offensive SBIC ToxiGen toxigen_new white_new
  )
  RUN_SPECS=()
  case "${_seed}" in
    # Any seed_N_model / _new / _wCL / _woCL ablation bucket (longer suffixes first).
    seed_*_model_woCL|seed_*_model_wCL|seed_*_model_new|seed_*_model)
      for _ds in "${_seed_all[@]}"; do
        RUN_SPECS+=("${_ds}|${_run_rel}/${_ds}/${_rb}|${_split}|none")
      done
      return 0
      ;;
    # Mech / GT-target buckets (seed_0_mech, …): include HateXplain label variants.
    seed_*_mech*)
      for _ds in IHC DynaHate HatEval HateXplain offensive_as_hate offensive_as_not_hate \
        offensive_as_not_hate_soft_none drop_offensive SBIC; do
        RUN_SPECS+=("${_ds}|${_run_rel}/${_ds}/${_rb}|${_split}|none")
      done
      return 0
      ;;
  esac
  if [[ "${_E}" == roberta_large ]]; then
    _bb="${_rb}"
  else
    _bb="bert_large_baseline"
  fi
  for _ds in IHC DynaHate HatEval HateXplain offensive_as_not_hate offensive_as_not_hate_soft_none; do
    RUN_SPECS+=("${_ds}|${_run_rel}/${_ds}/${_bb}|${_split}|none")
  done
}

_submit_mech_cue_validate_slurm_scripts() {
  local _mech="${SLURM_MECH:-job_submission/stage2/cue_mech_attnmask/slurm_mech_distill_single.sh}"
  case "${_mech}" in
    *slurm_cue_distill.sh|*slurm_train_cue_span*)
      echo "[submit_roberta_fft_mech_cue] ERROR: SLURM_MECH must be slurm_mech_distill_single.sh." >&2
      exit 1
      ;;
  esac
  if [[ ! -f "${LOCAL_ECHOHATE}/${_mech}" ]]; then
    echo "[submit_roberta_fft_mech_cue] ERROR: missing Slurm script: ${LOCAL_ECHOHATE}/${_mech}" >&2
    exit 1
  fi
  if [[ -n "${SLURM_CUE:-}" ]]; then
    echo "[submit_roberta_fft_mech_cue] ERROR: SLURM_CUE is set but cue training is not supported (${SLURM_CUE})." >&2
    echo "  Unset SLURM_CUE and use APPROACH=mech." >&2
    exit 1
  fi
}

_submit_mech_cue_describe_approach() {
  local _mech_n="${#MECH_GRID_ARR[@]}"
  local _einit_n="${#E_INIT_GRID_ARR[@]}"
  local _ainit_n="${#A_INIT_GRID_ARR[@]}"
  local _tinit_n="${#T_INIT_GRID_ARR[@]}"
  local _cue_lm_n="${#CUE_MECH_GRID_ARR[@]}"
  local _cue_n _scale_n
  if [[ "${E_SCALE_MODE}" == "split" ]]; then
    _scale_n=$(( _ainit_n * _tinit_n ))
  else
    _scale_n="${_einit_n}"
  fi
  if [[ "${CUE_LAMBDA_MECH_GRID:-}" == "0" ]]; then
    _cue_n="${_scale_n}"
  else
    _cue_n=$(( _cue_lm_n * _scale_n ))
  fi
  case "${APPROACH}" in
    mech)
      echo "[submit_roberta_fft_mech_cue] Mode: mech alone (no cue-span attn) → ${_mech_n} λ/dataset×teacher_mode$([ "${PACK_LAMBDA_GRID}" = "1" ] && echo " (PACK_LAMBDA_GRID=1 → 1 sbatch packs all λ)" || echo " (1 sbatch / λ)")"
      ;;
    cue)
      if [[ "${CUE_LAMBDA_MECH_GRID:-}" == "0" ]]; then
        if [[ "${E_SCALE_MODE}" == "split" ]]; then
          echo "[submit_roberta_fft_mech_cue] Mode: attn alone (split a×t) → ${_cue_n} jobs/dataset×teacher_mode (A_INIT_GRID × T_INIT_GRID, λ_mech=0)"
        else
          echo "[submit_roberta_fft_mech_cue] Mode: attn alone (no mech head) → ${_cue_n} jobs/dataset×teacher_mode (E_INIT_GRID, λ_mech=0)"
        fi
      else
        if [[ "${E_SCALE_MODE}" == "split" ]]; then
          echo "[submit_roberta_fft_mech_cue] Mode: full cue (split a×t + mech) → ${_cue_n} jobs/dataset×teacher_mode (CUE_LAMBDA_MECH_GRID × A × T)"
        else
          echo "[submit_roberta_fft_mech_cue] Mode: full cue (attn + mech) → ${_cue_n} jobs/dataset×teacher_mode (CUE_LAMBDA_MECH_GRID × E_INIT_GRID)"
        fi
      fi
      ;;
    both)
      echo "[submit_roberta_fft_mech_cue] Mode: both → mech (${_mech_n}) + cue (${_cue_n}) jobs/dataset×teacher_mode"
      ;;
  esac
  echo "[submit_roberta_fft_mech_cue] Slurm: mech=${SLURM_MECH}"
}

_apply_ihc_baseline_run_override() {
  _apply_dataset_baseline_run_override "IHC" "ENCODER_IHC_BASELINE_RUN_REL"
}

_apply_dynahate_baseline_run_override() {
  _apply_dataset_baseline_run_override "DynaHate" "ENCODER_DYNAHATE_BASELINE_RUN_REL"
}

# Replace or append RUN_SPEC for a single dataset (seed_0 HatEval / HateXplain).
_apply_dataset_baseline_run_override() {
  local _ds="$1"
  local _rel_var="$2"
  local _rel="${!_rel_var:-}"
  [[ -z "${_rel}" ]] && return 0
  _rel="${_rel#./}"
  # Ablation buckets: refuse teacher/baseline from a different seed folder (cross-CL contamination).
  case "${ECHOHATE_DATA_SEED_DIR:-}" in
    *_wCL|*_woCL)
      if [[ "${_rel}" != *"/${ECHOHATE_DATA_SEED_DIR}/"* && "${_rel}" != "${ECHOHATE_DATA_SEED_DIR}/"* && "${_rel}" != *"data/${ECHOHATE_DATA_SEED_DIR}/"* ]]; then
        # Allow relative paths that still embed the seed dir somewhere.
        if [[ "${_rel}" != *"${ECHOHATE_DATA_SEED_DIR}"* ]]; then
          echo "ERROR: ${_rel_var}=${_rel} does not contain seed bucket '${ECHOHATE_DATA_SEED_DIR}' (would mix wCL/woCL teachers)" >&2
          exit 1
        fi
      fi
      ;;
  esac
  local _new_specs=()
  local spec ds found=0
  for spec in "${RUN_SPECS[@]}"; do
    ds="${spec%%|*}"
    if [[ "${ds}" == "${_ds}" ]]; then
      found=1
      _new_specs+=("${_ds}|${_rel}|splitted_data_80_10_10|none")
    else
      _new_specs+=("${spec}")
    fi
  done
  if [[ "${found}" == "0" ]]; then
    _new_specs+=("${_ds}|${_rel}|splitted_data_80_10_10|none")
  fi
  RUN_SPECS=("${_new_specs[@]}")
}

_apply_hateval_baseline_run_override() {
  _apply_dataset_baseline_run_override "HatEval" "ENCODER_HATEVAL_BASELINE_RUN_REL"
}

_apply_hatexplain_baseline_run_override() {
  _apply_dataset_baseline_run_override "HateXplain" "ENCODER_HATEXPLAIN_BASELINE_RUN_REL"
}

_apply_ethos_baseline_run_override() {
  _apply_dataset_baseline_run_override "Ethos" "ENCODER_ETHOS_BASELINE_RUN_REL"
}


_apply_offensive_as_hate_baseline_run_override() {
  _apply_dataset_baseline_run_override "offensive_as_hate" "ENCODER_OFFENSIVE_AS_HATE_BASELINE_RUN_REL"
}

_apply_offensive_as_not_hate_baseline_run_override() {
  _apply_dataset_baseline_run_override "offensive_as_not_hate" "ENCODER_OFFENSIVE_AS_NOT_HATE_BASELINE_RUN_REL"
}

_apply_offensive_as_not_hate_soft_none_baseline_run_override() {
  _apply_dataset_baseline_run_override "offensive_as_not_hate_soft_none" "ENCODER_OFFENSIVE_AS_NOT_HATE_SOFT_NONE_BASELINE_RUN_REL"
}

_apply_drop_offensive_baseline_run_override() {
  _apply_dataset_baseline_run_override "drop_offensive" "ENCODER_DROP_OFFENSIVE_BASELINE_RUN_REL"
}

_apply_drop_offensive_repo_split_baseline_run_override() {
  _apply_dataset_baseline_run_override "drop_offensive_repo_split" "ENCODER_DROP_OFFENSIVE_REPO_SPLIT_BASELINE_RUN_REL"
}

_apply_sbic_baseline_run_override() {
  _apply_dataset_baseline_run_override "SBIC" "ENCODER_SBIC_BASELINE_RUN_REL"
}

_apply_toxigen_baseline_run_override() {
  # Folder on disk is toxigen_new; keep ToxiGen for legacy seed buckets that still list it.
  if [[ "${ONLY_DATASETS:-}" == *"toxigen_new"* ]] || [[ "${ECHOHATE_DATA_SEED_DIR:-}" == *"_new" ]]; then
    _apply_dataset_baseline_run_override "toxigen_new" "ENCODER_TOXIGEN_BASELINE_RUN_REL"
  else
    _apply_dataset_baseline_run_override "ToxiGen" "ENCODER_TOXIGEN_BASELINE_RUN_REL"
  fi
}

_apply_white_new_baseline_run_override() {
  _apply_dataset_baseline_run_override "white_new" "ENCODER_WHITE_NEW_BASELINE_RUN_REL"
}

# FFT checkpoint: …/roberta_large_baseline/model/pytorch_model.pt (or INIT_BACKBONE_FROM).
# Fallback: best HF Trainer checkpoint from metric_summary (checkpoint-N/), which holds model.safetensors.
# INIT_BACKBONE_FROM may be a file, Trainer ckpt dir, or a full run dir (attn/mech/baseline).
_metric_best_ckpt_hpc() {
  local _root="$1"
  _hpc_ssh "${HPC_USER}@${HPC_HOST}" "python3 - <<'PY'
import json
from pathlib import Path
p = Path('${_root}/metric_summary.json')
if not p.is_file():
    raise SystemExit(0)
j = json.loads(p.read_text())
be = ((j.get('training_config') or {}).get('best_epoch_by_eval_f1_macro') or {})
ck = be.get('best_model_checkpoint') or ''
print(ck.strip())
PY
" 2>/dev/null | tail -n1 | tr -d '\r' || true
}

_pick_init_backbone_hpc() {
  local run_root="$1"
  local _rel="${run_root#./}"
  _rel="${_rel%/}"
  local _base="${HPC_DIR}/${_rel}"
  local _candidates=()
  local _c _best

  if [[ -n "${INIT_BACKBONE_FROM:-}" ]]; then
    _c="${INIT_BACKBONE_FROM}"
    if [[ "${_c}" != /* ]]; then
      _c="${HPC_DIR}/${_c#./}"
    fi
    # Accept file, Trainer ckpt dir, baseline run, or cue/attn run dir (cue_span_state.pt).
    _candidates=(
      "${_c}"
      "${_c}/cue_span_state.pt"
      "${_c}/encoder_model.pt"
      "${_c}/model/pytorch_model.pt"
      "${_c}/pytorch_model.pt"
    )
    _best="$(_metric_best_ckpt_hpc "${_c}")"
    if [[ -n "${_best}" ]]; then
      _candidates+=("${_best}")
    fi
    # If INIT points at …/model/pytorch_model.pt, also probe sibling metric_summary.
    if [[ "${_c}" == */model/pytorch_model.pt ]]; then
      _best="$(_metric_best_ckpt_hpc "${_c%/model/pytorch_model.pt}")"
      if [[ -n "${_best}" ]]; then
        _candidates+=("${_best}")
      fi
    fi
    # If INIT points at …/cue_span_state.pt, also probe sibling Trainer best ckpt.
    if [[ "${_c}" == */cue_span_state.pt ]]; then
      _best="$(_metric_best_ckpt_hpc "${_c%/cue_span_state.pt}")"
      if [[ -n "${_best}" ]]; then
        _candidates+=("${_best}")
      fi
    fi
  elif [[ "${ENCODER_RUN_TAG}" == *warmft* ]] || [[ "${INIT_BACKBONE_AUTO:-0}" == "1" ]]; then
    _candidates=(
      "${_base}/model/pytorch_model.pt"
      "${_base}/pytorch_model.pt"
      "${_base}/cue_span_state.pt"
      "${_base}/encoder_model.pt"
    )
    _best="$(_metric_best_ckpt_hpc "${_base}")"
    if [[ -n "${_best}" ]]; then
      _candidates+=("${_best}")
    fi
  fi

  if [[ ${#_candidates[@]} -gt 0 ]]; then
    for _c in "${_candidates[@]}"; do
      if _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${_c}' || test -f '${_c}/cue_span_state.pt' || test -f '${_c}/encoder_model.pt' || test -f '${_c}/model.safetensors' || test -f '${_c}/pytorch_model.pt' || test -f '${_c}/pytorch_model.bin' || test -f '${_c}/model/pytorch_model.pt'"; then
        # Prefer concrete weight file when a run dir matched via nested cue_span_state.pt.
        if _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${_c}'"; then
          printf '%s' "${_c}"
        elif _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${_c}/cue_span_state.pt'"; then
          printf '%s' "${_c}/cue_span_state.pt"
        elif _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${_c}/encoder_model.pt'"; then
          printf '%s' "${_c}/encoder_model.pt"
        elif _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${_c}/model/pytorch_model.pt'"; then
          printf '%s' "${_c}/model/pytorch_model.pt"
        else
          printf '%s' "${_c}"
        fi
        return 0
      fi
    done
    printf '%s' "${_candidates[0]}"
  fi
}

# Resolve INIT_BACKBONE_FROM export fragment for sbatch (cue + mech). Prints fragment or nothing.
# Returns 1 when warmft is required but checkpoint missing (caller should skip/exit).
_warmft_sbatch_export_frag() {
  local run_root="$1"
  local _init_bb_hpc=""
  local _frag=""
  if [[ "${_WARMFT_MODE}" != "1" ]] && [[ -z "${INIT_BACKBONE_FROM:-}" ]] \
    && [[ "${ENCODER_RUN_TAG}" != *warmft* ]] && [[ "${INIT_BACKBONE_AUTO:-0}" != "1" ]]; then
    return 0
  fi
  _init_bb_hpc="$(_pick_init_backbone_hpc "${run_root}")"
  if [[ -z "${_init_bb_hpc}" ]]; then
    echo "ERROR: warmft mode but no INIT_BACKBONE_FROM resolved for run_root=${run_root}" >&2
    return 1
  fi
  if ! _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${_init_bb_hpc}' || test -f '${_init_bb_hpc}/cue_span_state.pt' || test -f '${_init_bb_hpc}/encoder_model.pt' || test -f '${_init_bb_hpc}/model.safetensors' || test -f '${_init_bb_hpc}/pytorch_model.pt' || test -f '${_init_bb_hpc}/pytorch_model.bin' || test -f '${_init_bb_hpc}/model/pytorch_model.pt'"; then
    echo "ERROR: warm-start checkpoint missing on HPC: ${_init_bb_hpc}" >&2
    echo "  Tried cue_span_state.pt / encoder_model.pt / model/pytorch_model.pt and best checkpoint from metric_summary." >&2
    return 1
  fi
  # Normalize run-dir INIT to concrete weight file for sbatch.
  if _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -d '${_init_bb_hpc}' && test -f '${_init_bb_hpc}/cue_span_state.pt'"; then
    _init_bb_hpc="${_init_bb_hpc}/cue_span_state.pt"
  elif _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -d '${_init_bb_hpc}' && test -f '${_init_bb_hpc}/encoder_model.pt'"; then
    _init_bb_hpc="${_init_bb_hpc}/encoder_model.pt"
  elif _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -d '${_init_bb_hpc}' && test -f '${_init_bb_hpc}/model/pytorch_model.pt'"; then
    _init_bb_hpc="${_init_bb_hpc}/model/pytorch_model.pt"
  fi
  _frag=",INIT_BACKBONE_FROM=${_init_bb_hpc}"
  if [[ "${INIT_LOAD_CLASSIFIER:-1}" == "0" ]]; then
    _frag="${_frag},INIT_LOAD_CLASSIFIER=0"
  else
    _frag="${_frag},INIT_LOAD_CLASSIFIER=1"
  fi
  if [[ "${INIT_LOAD_E:-1}" == "0" ]]; then
    _frag="${_frag},INIT_LOAD_E=0"
  else
    _frag="${_frag},INIT_LOAD_E=1"
  fi
  if [[ -n "${FREEZE_BACKBONE_RATIO:-}" && "${FREEZE_BACKBONE_RATIO}" != "0" && "${FREEZE_BACKBONE_RATIO}" != "0.0" ]]; then
    _frag="${_frag},FREEZE_BACKBONE_RATIO=${FREEZE_BACKBONE_RATIO}"
    _frag="${_frag},FREEZE_SWAP_MODE=${FREEZE_SWAP_MODE:-head_then_joint}"
    # Use '+' not ',' — commas break Slurm --export=ALL,K=v1,v2 parsing.
    _fs_mods="${FREEZE_PHASE1_MODULES:-e,mechanism_head,classifier}"
    _fs_mods="${_fs_mods//,/+}"
    _frag="${_frag},FREEZE_PHASE1_MODULES=${_fs_mods}"
  fi
  echo "  INIT_BACKBONE_FROM=${_init_bb_hpc}  INIT_LOAD_CLASSIFIER=${INIT_LOAD_CLASSIFIER:-1}  INIT_LOAD_E=${INIT_LOAD_E:-1}" >&2
  if [[ -n "${FREEZE_BACKBONE_RATIO:-}" && "${FREEZE_BACKBONE_RATIO}" != "0" && "${FREEZE_BACKBONE_RATIO}" != "0.0" ]]; then
    echo "  FREEZE_BACKBONE_RATIO=${FREEZE_BACKBONE_RATIO}  FREEZE_SWAP_MODE=${FREEZE_SWAP_MODE:-head_then_joint}" >&2
  fi
  printf '%s' "${_frag}"
  return 0
}

_assert_explicit_init_backbone_single_dataset() {
  [[ -z "${INIT_BACKBONE_FROM:-}" ]] && return 0
  local spec ds
  local _seen_csv=""
  local _n=0
  for spec in "${RUN_SPECS[@]}"; do
    ds="${spec%%|*}"
    if ! _spec_ok "${ds}"; then
      continue
    fi
    case ",${_seen_csv}," in
      *",${ds},"*) continue ;;
    esac
    _seen_csv="${_seen_csv:+${_seen_csv},}${ds}"
    _n=$((_n + 1))
  done
  if [[ "${_n}" -gt 1 ]]; then
    echo "ERROR: INIT_BACKBONE_FROM is a single path but ${_n} datasets would share it:" >&2
    echo "  ${INIT_BACKBONE_FROM}" >&2
    echo "Unset INIT_BACKBONE_FROM and use ENCODER_RUN_TAG=_warmft (auto per-dataset baseline)," >&2
    echo "or pass a single --dataset with an explicit INIT_BACKBONE_FROM." >&2
    exit 1
  fi
}

# RUN_SPECS is built per submit wave via _refresh_run_specs_array (uses current VALIDATION_ROOT_ENCODER_BN / _E).

IFS=',' read -r -a MODE_ARR <<< "${TEACHER_MODES}"
IFS=',' read -r -a MECH_GRID_ARR <<< "${LAMBDA_MECH_GRID}"
IFS=',' read -r -a E_INIT_GRID_ARR <<< "${E_INIT_GRID}"
IFS=',' read -r -a CUE_MECH_GRID_ARR <<< "${CUE_LAMBDA_MECH_GRID}"
IFS=',' read -r -a TARGET_GRID_ARR <<< "${LAMBDA_TARGET_GRID}"
A_INIT_GRID_ARR=()
T_INIT_GRID_ARR=()
if [[ "${E_SCALE_MODE}" == "split" ]]; then
  IFS=',' read -r -a A_INIT_GRID_ARR <<< "${A_INIT_GRID}"
  IFS=',' read -r -a T_INIT_GRID_ARR <<< "${T_INIT_GRID}"
  if [[ ${#A_INIT_GRID_ARR[@]} -lt 1 || ${#T_INIT_GRID_ARR[@]} -lt 1 ]]; then
    echo "ERROR: E_SCALE_MODE=split requires non-empty A_INIT_GRID and T_INIT_GRID" >&2
    exit 1
  fi
fi

_submit_mech_cue_validate_slurm_scripts
_submit_mech_cue_describe_approach

SUBMIT_SBATCH_OK=0
SUBMIT_SBATCH_SKIPPED_EXISTING=0
SUBMIT_SKIPPED_MISSING_INPUTS=0

if [[ "${TEACHER_MODES_ORDER}" == "nocontext_first" ]]; then
  SUBMIT_PHASE_LIST=(nocontext traincontext)
else
  SUBMIT_PHASE_LIST=(all)
fi

_SUBMIT_PROFILE_WAVES=()
if [[ -n "${SUBMIT_STUDENT_TEACHER_PROFILES:-}" ]]; then
  IFS=',' read -r -a _spf_parse <<< "${SUBMIT_STUDENT_TEACHER_PROFILES}"
  for _p in "${_spf_parse[@]}"; do
    _p="$(echo "${_p}" | xargs)"
    [[ -z "${_p}" ]] && continue
    if [[ "${_p}" != *"|"* ]]; then
      echo "ERROR: SUBMIT_STUDENT_TEACHER_PROFILES entries must be encoder_bn|lm_family (e.g. bert_large|qwen), got '${_p}'" >&2
      exit 1
    fi
    _SUBMIT_PROFILE_WAVES+=("${_p}")
  done
  if [[ ${#_SUBMIT_PROFILE_WAVES[@]} -lt 1 ]]; then
    echo "ERROR: SUBMIT_STUDENT_TEACHER_PROFILES set but no valid entries after parse" >&2
    exit 1
  fi
else
  _SUBMIT_PROFILE_WAVES=("DEFAULT")
fi

for _spf_wave in "${_SUBMIT_PROFILE_WAVES[@]}"; do
  if [[ "${_spf_wave}" == "DEFAULT" ]]; then
    TEACHER_LM_FAMILY="${_USER_TEACHER_LM_FAMILY:-}"
    TEACHER_OUTPUT_TAG="${_BASE_TEACHER_OUTPUT_TAG:-}"
    TEACHER_OUTPUT_TAG_OVERRIDES="${_BASE_TEACHER_OUTPUT_TAG_OVERRIDES:-}"
    if [[ -n "${TEACHER_LM_FAMILY}" ]]; then
      _apply_llm_vars_for_lm_family "${TEACHER_LM_FAMILY}"
    else
      _merge_bert_phi_tag_overrides
    fi
  else
    _enc_bn="${_spf_wave%%|*}"
    _lm_pf="${_spf_wave#*|}"
    _enc_bn="$(echo "${_enc_bn}" | xargs)"
    _lm_pf="$(echo "${_lm_pf}" | xargs)"
    _set_encoder_encoder_from_bn_name "${_enc_bn}"
    _apply_llm_vars_for_lm_family "${_lm_pf}"
  fi
  TEACHER_PATH_SLUG="$(echo "${TEACHER_LM_FAMILY:-phi}" | tr '[:upper:]' '[:lower:]')"
  [[ -z "${TEACHER_PATH_SLUG}" ]] && TEACHER_PATH_SLUG="phi"

  CURRENT_LLM_OUTPUT_TAG="${TEACHER_OUTPUT_TAG:-}"
  _refresh_run_specs_array
  _apply_ihc_baseline_run_override
  _apply_dynahate_baseline_run_override
  _apply_hateval_baseline_run_override
  _apply_hatexplain_baseline_run_override
  _apply_ethos_baseline_run_override
  _apply_offensive_as_hate_baseline_run_override
  _apply_offensive_as_not_hate_baseline_run_override
  _apply_offensive_as_not_hate_soft_none_baseline_run_override
  _apply_drop_offensive_baseline_run_override
  _apply_drop_offensive_repo_split_baseline_run_override
  _apply_sbic_baseline_run_override
  _apply_toxigen_baseline_run_override
  _apply_white_new_baseline_run_override
  _assert_explicit_init_backbone_single_dataset

  echo "=========================================="
  echo "[submit_roberta_fft_mech_cue] Submit wave: VALIDATION_ROOT_ENCODER_BN=${_E} MODEL_NAME=${MODEL_NAME} TEACHER_LM_FAMILY=${TEACHER_LM_FAMILY:-?} TEACHER_PATH_SLUG=${TEACHER_PATH_SLUG}"
  echo "  TEACHER_OUTPUT_TAG=${TEACHER_OUTPUT_TAG:-<empty>}  TEACHER_OUTPUT_TAG_OVERRIDES=${TEACHER_OUTPUT_TAG_OVERRIDES:-<none>}"
  echo "=========================================="
  echo "Dataset pos_weight map (${_E}):"
  for spec in "${RUN_SPECS[@]}"; do
    IFS='|' read -r ds _run _split posw out_variant <<< "${spec}"
    out_variant="$(echo "${out_variant:-}" | xargs)"
    if [[ -n "${out_variant}" ]]; then
      echo "  ${ds}/${out_variant}: ${posw}"
    else
      echo "  ${ds}: ${posw}"
    fi
  done
  echo "=========================================="

for _submit_phase in "${SUBMIT_PHASE_LIST[@]}"; do
  if [[ "${TEACHER_MODES_ORDER}" == "nocontext_first" ]]; then
    echo "=========================================="
    echo "llm wave: phase=${_submit_phase}  (nocontext_first: nocontext jobs before traincontext)"
    echo "=========================================="
  fi

for spec in "${RUN_SPECS[@]}"; do
  IFS='|' read -r ds run_root split_sub posw out_variant <<< "${spec}"
  out_variant="$(echo "${out_variant:-}" | xargs)"
  if [[ -n "${out_variant}" ]]; then
    ENCODER_OUT_BASE="${ENCODER_DATA_ROOT}/${ds}/${out_variant}"
    JOB_DS_TAG="${ds}_${out_variant}"
  else
    ENCODER_OUT_BASE="${ENCODER_DATA_ROOT}/${ds}"
    JOB_DS_TAG="${ds}"
  fi
  if [[ "${ENCODER_FFT_OUT_FLAT}" != "1" ]]; then
    # Optional override: full baseline folder bn (e.g. roberta_large_baseline_wCL) so
    # eacl_seed0 wCL/woCL do not share …/roberta_large_encoder/.
    if [[ -n "${ENCODER_OUT_DIR_BN:-}" ]]; then
      ENCODER_OUT_BASE="${ENCODER_OUT_BASE}/${ENCODER_OUT_DIR_BN}_encoder"
    else
      ENCODER_OUT_BASE="${ENCODER_OUT_BASE}/${_E}_encoder"
    fi
  fi
  if ! _spec_ok "${ds}"; then
    continue
  fi
  if [[ "${ds}" == "Ethos" ]] && [[ -n "${out_variant}" ]] && ! _ethos_variant_ok "${out_variant}"; then
    continue
  fi
  resolve_current_llm_output_tag "${JOB_DS_TAG}" "${ds}"
  if [[ -n "${TEACHER_OUTPUT_TAG_OVERRIDES:-}" ]]; then
    echo "[submit_roberta_fft_mech_cue] ${JOB_DS_TAG}: llm dirname suffix='${CURRENT_LLM_OUTPUT_TAG:-<empty>}'"
  fi
  for mode in "${MODE_ARR[@]}"; do
    mode="$(_normalize_teacher_mode "$(echo "${mode}" | xargs)")"
    [[ -z "${mode}" ]] && continue
    if [[ "${TEACHER_MODES_ORDER}" == "nocontext_first" ]]; then
      if [[ "${_submit_phase}" == "nocontext" ]] && [[ "${mode}" != "nocontext" ]]; then
        continue
      fi
      if [[ "${_submit_phase}" == "traincontext" ]] && [[ "${mode}" == "nocontext" ]]; then
        continue
      fi
    fi
    tdir="$(llm_dir_for_mode "${mode}")"
    llm_rel="${run_root}/${tdir}/selected_k_llm_predictions.csv"
    llm_rel_trainid="${run_root}/${tdir}/selected_k_llm_predictions_trainid.csv"
    llm_hpc_sel="${HPC_DIR}/${llm_rel}"
    llm_hpc_trainid="${HPC_DIR}/${llm_rel_trainid}"
    # Optional: teacher lives under a mech-only (or other) run, not under encoder llm_* .
    if [[ -n "${LLM_TEACHER_DIR:-}" ]]; then
      # Expand {ds} so one template covers multi-dataset submits.
      _ltd="${LLM_TEACHER_DIR//\{ds\}/${ds}}"
      if [[ "${_ltd}" != /* ]]; then
        _ltd="${HPC_DIR}/${_ltd#./}"
      fi
      # Accept either the llm_* folder or its parent run dir.
      if _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${_ltd}/selected_k_llm_predictions.csv'"; then
        :
      elif _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${_ltd}/${tdir}/selected_k_llm_predictions.csv'"; then
        _ltd="${_ltd}/${tdir}"
      fi
      llm_hpc_sel="${_ltd}/selected_k_llm_predictions.csv"
      llm_hpc_trainid="${_ltd}/selected_k_llm_predictions_trainid.csv"
      echo "[submit_roberta_fft_mech_cue] LLM_TEACHER_DIR → ${_ltd}"
    fi
    llm_hpc="${llm_hpc_sel}"
    train_csv_hpc="${ENCODER_DATA_ROOT}/${ds}/${split_sub}/train.csv"
    validation_csv_hpc="${ENCODER_DATA_ROOT}/${ds}/${split_sub}/validation.csv"
    run_tune_hpc="${HPC_DIR}/${run_root}"
    if [[ "${AUTO_FIX_TRAINID_FROM_TEXT}" == "1" ]]; then
      if ! _hpc_ssh "${HPC_USER}@${HPC_HOST}" "cd '${HPC_DIR}' && set +u; source job_submission/hpc_env_deakin.sh >/dev/null 2>&1; set -u; python - <<'PY'
import pandas as pd
from pathlib import Path

src = Path('${llm_hpc}')
dst = Path('${llm_hpc_trainid}')
train_csv = Path('${train_csv_hpc}')
if (not src.is_file()) or (not train_csv.is_file()):
    raise SystemExit(0)

needs_repair = True
if dst.is_file():
    try:
        old = pd.read_csv(dst, usecols=['row_id'])
        s = old['row_id'].astype(str)
        needs_repair = (len(s) == 0) or s.str.startswith('train_combined_').any()
    except Exception:
        needs_repair = True
if not needs_repair:
    raise SystemExit(0)

df = pd.read_csv(src)
tr = pd.read_csv(train_csv)
if 'text' not in df.columns or 'text' not in tr.columns:
    raise SystemExit(0)

def norm_text(v):
    return ' '.join(str(v).strip().lower().split())

train_ids = tr['row_id'].astype(str).tolist() if 'row_id' in tr.columns else [f'train_{i}' for i in range(len(tr))]
trn = tr.copy()
trn['__norm_text'] = trn['text'].map(norm_text)
trn['__train_row_id'] = train_ids
uniq_mask = trn.groupby('__norm_text')['__train_row_id'].transform('nunique') == 1
map_df = trn.loc[uniq_mask, ['__norm_text', '__train_row_id']].drop_duplicates('__norm_text')
text_to_train = dict(zip(map_df['__norm_text'], map_df['__train_row_id']))
out = df.copy()
out['__norm_text'] = out['text'].map(norm_text)
out['row_id'] = out['__norm_text'].map(text_to_train)
matched_text = int(out['row_id'].notna().sum())
if 'row_id' in df.columns:
    orig = df['row_id'].astype(str)
    # Deterministic fallback for train_combined_<idx> ids: map by split-train row position
    # only when normalized text at that index matches (safety against index drift).
    idx = orig.str.extract(r'^train_combined_(\d+)$')[0]
    idx_vals = pd.to_numeric(idx, errors='coerce')
    idx_cand = out['row_id'].isna() & idx_vals.notna()
    idx_inrange = idx_cand & (idx_vals >= 0) & (idx_vals < len(tr))
    idx_ok2 = idx_inrange.copy()
    idx_text_mismatch = 0
    if idx_inrange.any():
        pos_arr = idx_vals[idx_inrange].astype(int).to_numpy()
        llm_norm = out.loc[idx_inrange, '__norm_text'].astype(str).to_numpy()
        trn_norm = trn.iloc[pos_arr]['__norm_text'].astype(str).to_numpy()
        matched_by_pos = (llm_norm == trn_norm)
        idx_ok2.loc[idx_inrange] = matched_by_pos
        idx_text_mismatch = int((~matched_by_pos).sum())
    if idx_ok2.any():
        pos = idx[idx_ok2].astype(int).tolist()
        out.loc[idx_ok2, 'row_id'] = [train_ids[i] for i in pos]
    matched_index = int(idx_ok2.sum())
    # Keep original id only when it is already split-style.
    keep_orig_mask = out['row_id'].isna() & (~orig.str.startswith('train_combined_'))
    out.loc[keep_orig_mask, 'row_id'] = orig.loc[keep_orig_mask]
    matched_orig = int(keep_orig_mask.sum())
else:
    matched_index = 0
    matched_orig = 0
    idx_text_mismatch = 0
out = out.drop(columns=['__norm_text'])
matched = int(out['row_id'].notna().sum())
total = len(out)
dst.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dst, index=False)
print(
    f'AUTO_FIX_TRAINID wrote {dst} rows={total} matched={matched} unmatched={total-matched} '
    f'(text={matched_text}, index={matched_index}, orig={matched_orig}, index_text_mismatch={idx_text_mismatch})'
)
PY"; then
        echo "WARN: AUTO_FIX_TRAINID failed for ${JOB_DS_TAG} ${mode}; continuing with existing trainid file." >&2
      fi
    fi
    if [[ "${CHECK_TEACHER_AT_SUBMIT}" == "1" ]]; then
      if _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${llm_hpc_trainid}'"; then
        llm_hpc="${llm_hpc_trainid}"
        echo "Using trainid llm file: ${llm_hpc}"
      fi
      if ! _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${llm_hpc}'"; then
        echo "SKIP ${JOB_DS_TAG} ${mode}: missing ${llm_hpc}"
        continue
      fi
    else
      # Defer llm file validation to the slurm job preflight (prefer trainid when present).
      llm_hpc="${llm_hpc_trainid}"
    fi

    if [[ "${CHECK_INPUTS_AT_SUBMIT}" == "1" ]]; then
      if _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${llm_hpc_trainid}'"; then
        llm_hpc="${llm_hpc_trainid}"
      elif _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${llm_hpc_sel}'"; then
        llm_hpc="${llm_hpc_sel}"
        echo "Using selected_k llm file (no trainid on HPC): ${llm_hpc}"
      fi
    fi

    if [[ "${CHECK_INPUTS_AT_SUBMIT}" == "1" ]]; then
      if ! _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -f '${llm_hpc}' && test -f '${train_csv_hpc}' && test -f '${validation_csv_hpc}' && test -d '${run_tune_hpc}'"; then
        echo "SKIP ${JOB_DS_TAG} ${mode}: HPC input preflight failed (need all of the following on the cluster):" >&2
        echo "  llm=${llm_hpc}" >&2
        echo "  train_csv=${train_csv_hpc}" >&2
        echo "  validation=${validation_csv_hpc}" >&2
        echo "  run_dir=${run_tune_hpc}" >&2
        SUBMIT_SKIPPED_MISSING_INPUTS=$((SUBMIT_SKIPPED_MISSING_INPUTS + 1))
        if [[ "${CHECK_INPUTS_STRICT}" == "1" ]]; then
          echo "ERROR: CHECK_INPUTS_STRICT=1 and required inputs missing for ${JOB_DS_TAG} ${mode}" >&2
          exit 1
        fi
        continue
      fi
    fi

    if [[ "${APPROACH}" == "mech" || "${APPROACH}" == "both" ]]; then
      if [[ "${PACK_LAMBDA_GRID}" == "1" ]]; then
        # One sbatch per λ_target packs all λ_mech (plus-sep list; φ/qwen still separate).
        for lambda_target in "${TARGET_GRID_ARR[@]}"; do
          lambda_target="$(echo "${lambda_target}" | xargs)"
          [[ -z "${lambda_target}" ]] && continue
          lt_tag="${lambda_target/./p}"
          _lt_suffix=""
          if [[ "${lambda_target}" != "0" && "${lambda_target}" != "0.0" ]]; then
            _lt_suffix="_lt${lt_tag}"
          fi
          _pack_lms=()
          _pack_n_skip=0
          for lambda_mech in "${MECH_GRID_ARR[@]}"; do
            lambda_mech="$(echo "${lambda_mech}" | xargs)"
            [[ -z "${lambda_mech}" ]] && continue
            lm_tag="${lambda_mech/./p}"
            out_mech="${ENCODER_OUT_BASE}/encoder_fft_mech_seed${RANDOM_STATE}_${TEACHER_PATH_SLUG}_${mode}_lm${lm_tag}${_lt_suffix}${STUDENT_OUT_SUFFIX}"
            if [[ "${SKIP_IF_OUTPUT_EXISTS}" == "1" ]]; then
              if _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -s '${out_mech}/encoder_model.pt'"; then
                echo "PACK skip existing λ=${lambda_mech}: ${out_mech}"
                _pack_n_skip=$((_pack_n_skip + 1))
                SUBMIT_SBATCH_SKIPPED_EXISTING=$((SUBMIT_SBATCH_SKIPPED_EXISTING + 1))
                continue
              fi
            fi
            _pack_lms+=("${lambda_mech}")
          done
          if [[ ${#_pack_lms[@]} -eq 0 ]]; then
            echo "PACK skip (all λ already have encoder_model.pt): ${JOB_DS_TAG} ${mode} lt=${lambda_target}"
            continue
          fi
          # plus-join (commas break Slurm --export)
          _lm_list="${_pack_lms[0]}"
          for ((_pi=1; _pi<${#_pack_lms[@]}; _pi++)); do
            _lm_list="${_lm_list}+${_pack_lms[_pi]}"
          done
          _mech_prefix="${ENCODER_OUT_BASE}/encoder_fft_mech_seed${RANDOM_STATE}_${TEACHER_PATH_SLUG}_${mode}_"
          _mech_suffix="${_lt_suffix}${STUDENT_OUT_SUFFIX}"
          _n_pack=${#_pack_lms[@]}
          _pack_time="${PACK_SBATCH_TIME:-}"
          if [[ -z "${_pack_time}" ]]; then
            # Scale default 24h × n_λ, cap 72h.
            _hours=$((24 * _n_pack))
            if [[ ${_hours} -gt 72 ]]; then _hours=72; fi
            _pack_time="${_hours}:00:00"
          fi
          node="$(_pick_sbatch_nodelist)"
          sb_common="--partition=${SBATCH_PARTITION} --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=${SBATCH_CPUS} --mem=${SBATCH_MEM} --time=${_pack_time} --nodelist=${node}"
          _init_bb_mech=""
          if ! _init_bb_mech="$(_warmft_sbatch_export_frag "${run_root}")"; then
            if [[ "${CHECK_INPUTS_STRICT}" == "1" ]]; then
              exit 1
            fi
            SUBMIT_SKIPPED_MISSING_INPUTS=$((SUBMIT_SKIPPED_MISSING_INPUTS + 1))
            continue
          fi
          exp_mech="ALL,CONDA_ENV=${CONDA_ENV},ECHOHATE_SPLITTED_DATA_SUBDIR=${split_sub},DATA_ROOT=${ENCODER_DATA_ROOT},DATASET=${ds},MODEL_NAME=${MODEL_NAME},POS_WEIGHT_MODE=${posw},LLM_FILE=${llm_hpc},NUM_EPOCHS=${NUM_EPOCHS},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},LEARNING_RATE=${LEARNING_RATE},WEIGHT_DECAY=${WEIGHT_DECAY},WARMUP_RATIO=${WARMUP_RATIO},MAX_LENGTH=${MAX_LENGTH},RANDOM_STATE=${RANDOM_STATE},CLASS_WEIGHT_TARGET=${CLASS_WEIGHT_TARGET},CE_CONTRASTIVE_LAMBDA=${CE_CONTRASTIVE_LAMBDA},CONTRASTIVE_MARGIN=${CONTRASTIVE_MARGIN},LAMBDA_MECH_LIST=${_lm_list},MECH_OUT_DIR_PREFIX=${_mech_prefix},MECH_OUT_DIR_SUFFIX=${_mech_suffix},LAMBDA_MECH_RAMP_RATIO=0.1,EVALUATION_SPLIT=validation,RUN_EVAL_OVERWRITE_OUTPUTS=${RUN_EVAL_OVERWRITE_OUTPUTS},FORCE_RETRAIN=${FORCE_RETRAIN},TRACK_DYNAMICS=${TRACK_DYNAMICS},HATE_MECH_TAXONOMY=${HATE_MECH_TAXONOMY},ALLOW_TAXONOMY_OVERWRITE=${ALLOW_TAXONOMY_OVERWRITE:-0},MECH_AUX_MODE=${MECH_AUX_MODE:-mechanism}${_init_bb_mech}"
          jm="${TEACHER_PATH_SLUG}_fft_mech_${_E}_${JOB_DS_TAG}_${mode}_pack${_n_pack}lm${_lt_suffix}${STUDENT_OUT_SUFFIX}"
          echo "Submit MECH PACK ${JOB_DS_TAG} ${mode} n_λ=${_n_pack} (skip_existing=${_pack_n_skip}) list=${_lm_list} lt=${lambda_target} hate_mech=${HATE_MECH_TAXONOMY} mech_aux=${MECH_AUX_MODE:-mechanism} time=${_pack_time} -> nodelist=${node}"
          echo "  prefix=${_mech_prefix}  suffix=${_mech_suffix}"
          hpc_ssh_sbatch_export_all "${HPC_DIR}" "${SLURM_MECH}" "${exp_mech}" "--job-name=${jm} --output=${jm}_%j.out --error=${jm}_%j.err ${sb_common}" "unset CUDA_VISIBLE_DEVICES" || exit 1
          SUBMIT_SBATCH_OK=$((SUBMIT_SBATCH_OK + 1))
          echo "  JOB_ID=${HPC_LAST_SSH_JOB_ID}"
        done
      else
        for lambda_mech in "${MECH_GRID_ARR[@]}"; do
          lambda_mech="$(echo "${lambda_mech}" | xargs)"
          [[ -z "${lambda_mech}" ]] && continue
          lm_tag="${lambda_mech/./p}"
          for lambda_target in "${TARGET_GRID_ARR[@]}"; do
            lambda_target="$(echo "${lambda_target}" | xargs)"
            [[ -z "${lambda_target}" ]] && continue
            lt_tag="${lambda_target/./p}"
            _lt_suffix=""
            if [[ "${lambda_target}" != "0" && "${lambda_target}" != "0.0" ]]; then
              _lt_suffix="_lt${lt_tag}"
            fi
            out_mech="${ENCODER_OUT_BASE}/encoder_fft_mech_seed${RANDOM_STATE}_${TEACHER_PATH_SLUG}_${mode}_lm${lm_tag}${_lt_suffix}${STUDENT_OUT_SUFFIX}"
            _mech_skip_ok=0
            if [[ "${SKIP_IF_OUTPUT_EXISTS}" == "1" ]]; then
              # Skip when checkpoint already exists on HPC (file presence only; no login-node python).
              if _hpc_ssh "${HPC_USER}@${HPC_HOST}" "test -s '${out_mech}/encoder_model.pt'"; then
                _mech_skip_ok=1
              fi
            fi
            if [[ "${_mech_skip_ok}" == "1" ]]; then
              echo "SKIP MECH (encoder_model.pt exists on HPC): ${out_mech}"
              SUBMIT_SBATCH_SKIPPED_EXISTING=$((SUBMIT_SBATCH_SKIPPED_EXISTING + 1))
              continue
            fi
            node="$(_pick_sbatch_nodelist)"
            sb_common="--partition=${SBATCH_PARTITION} --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=${SBATCH_CPUS} --mem=${SBATCH_MEM} --time=${SBATCH_TIME} --nodelist=${node}"
            _init_bb_mech=""
            if ! _init_bb_mech="$(_warmft_sbatch_export_frag "${run_root}")"; then
              if [[ "${CHECK_INPUTS_STRICT}" == "1" ]]; then
                exit 1
              fi
              SUBMIT_SKIPPED_MISSING_INPUTS=$((SUBMIT_SKIPPED_MISSING_INPUTS + 1))
              continue
            fi
            exp_mech="ALL,CONDA_ENV=${CONDA_ENV},ECHOHATE_SPLITTED_DATA_SUBDIR=${split_sub},DATA_ROOT=${ENCODER_DATA_ROOT},DATASET=${ds},MODEL_NAME=${MODEL_NAME},POS_WEIGHT_MODE=${posw},LLM_FILE=${llm_hpc},OUTPUT_DIR=${out_mech},NUM_EPOCHS=${NUM_EPOCHS},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},LEARNING_RATE=${LEARNING_RATE},WEIGHT_DECAY=${WEIGHT_DECAY},WARMUP_RATIO=${WARMUP_RATIO},MAX_LENGTH=${MAX_LENGTH},RANDOM_STATE=${RANDOM_STATE},CLASS_WEIGHT_TARGET=${CLASS_WEIGHT_TARGET},CE_CONTRASTIVE_LAMBDA=${CE_CONTRASTIVE_LAMBDA},CONTRASTIVE_MARGIN=${CONTRASTIVE_MARGIN},LAMBDA_MECH=${lambda_mech},LAMBDA_MECH_RAMP_RATIO=0.1,EVALUATION_SPLIT=validation,RUN_EVAL_OVERWRITE_OUTPUTS=${RUN_EVAL_OVERWRITE_OUTPUTS},FORCE_RETRAIN=${FORCE_RETRAIN},TRACK_DYNAMICS=${TRACK_DYNAMICS},HATE_MECH_TAXONOMY=${HATE_MECH_TAXONOMY},ALLOW_TAXONOMY_OVERWRITE=${ALLOW_TAXONOMY_OVERWRITE:-0},MECH_AUX_MODE=${MECH_AUX_MODE:-mechanism}${_init_bb_mech}"
            jm="${TEACHER_PATH_SLUG}_fft_mech_${_E}_${JOB_DS_TAG}_${mode}_lm${lm_tag}${_lt_suffix}${STUDENT_OUT_SUFFIX}"
            echo "Submit MECH ${JOB_DS_TAG} ${mode} lm=${lambda_mech} lt=${lambda_target} hate_mech=${HATE_MECH_TAXONOMY} mech_aux=${MECH_AUX_MODE:-mechanism} -> nodelist=${node}  out=${out_mech}"
            hpc_ssh_sbatch_export_all "${HPC_DIR}" "${SLURM_MECH}" "${exp_mech}" "--job-name=${jm} --output=${jm}_%j.out --error=${jm}_%j.err ${sb_common}" "unset CUDA_VISIBLE_DEVICES" || exit 1
            SUBMIT_SBATCH_OK=$((SUBMIT_SBATCH_OK + 1))
            echo "  JOB_ID=${HPC_LAST_SSH_JOB_ID}"
          done
        done
      fi
    fi

    if [[ "${APPROACH}" == "cue" || "${APPROACH}" == "both" ]]; then
      echo "[submit_roberta_fft_mech_cue] ERROR: APPROACH=${APPROACH} not supported." >&2
      exit 1
    fi
    if false; then
      # Archived cue submit body retained only as unreachable historical reference.
      # Build scale cells: shared → one e; split → a×t. Each cell: dir_tag|e_init|a_init|t_init
      _CUE_SCALE_CELLS=()
      if [[ "${E_SCALE_MODE}" == "split" ]]; then
        for a_init in "${A_INIT_GRID_ARR[@]}"; do
          a_init="$(echo "${a_init}" | xargs)"
          [[ -z "${a_init}" ]] && continue
          a_tag="${a_init/./p}"
          for t_init in "${T_INIT_GRID_ARR[@]}"; do
            t_init="$(echo "${t_init}" | xargs)"
            [[ -z "${t_init}" ]] && continue
            t_tag="${t_init/./p}"
            _CUE_SCALE_CELLS+=("a${a_tag}_t${t_tag}|${a_init}|${a_init}|${t_init}")
          done
        done
      else
        for e_init in "${E_INIT_GRID_ARR[@]}"; do
          e_init="$(echo "${e_init}" | xargs)"
          [[ -z "${e_init}" ]] && continue
          e_tag="${e_init/./p}"
          _CUE_SCALE_CELLS+=("e${e_tag}|${e_init}||")
        done
      fi
      for cue_lambda_mech in "${CUE_MECH_GRID_ARR[@]}"; do
        cue_lambda_mech="$(echo "${cue_lambda_mech}" | xargs)"
        [[ -z "${cue_lambda_mech}" ]] && continue
        clm_tag="${cue_lambda_mech/./p}"
        for _cell in "${_CUE_SCALE_CELLS[@]}"; do
          IFS='|' read -r scale_tag e_init a_init t_init <<< "${_cell}"
          out_cue="${ENCODER_OUT_BASE}/encoder_fft_cue_span_seed${RANDOM_STATE}_${TEACHER_PATH_SLUG}_${mode}_${scale_tag}_lm${clm_tag}${STUDENT_OUT_SUFFIX}"
          _amp="${AMP_POOL:-sum}"
          if [[ "${_amp}" != "sum" ]]; then
            out_cue="${out_cue}_amp${_amp}"
          fi
          _cue_skip_test="test -s '${out_cue}/cue_span_state.pt' && test -s '${out_cue}/cue_span_config.json' && (test -s '${out_cue}/tokenizer/tokenizer.json' || test -s '${out_cue}/tokenizer/config.json' || test -s '${out_cue}/tokenizer/vocab.json')"
          if [[ "${RUN_FINAL_TEST_EVAL:-1}" == "1" ]] || [[ "${RUN_FINAL_TEST_EVAL:-1}" == "true" ]]; then
            _cue_skip_test="${_cue_skip_test} && test -s '${out_cue}/final_test_metric_summary_test.json' && test -s '${out_cue}/final_test_predictions_test.csv'"
          fi
          # Ampvaltest jobs are incomplete until valtest_amplified_* exist (do not skip on final_test alone).
          if [[ "${AMPVALTEST:-0}" == "1" || "${AMPVALTEST:-0}" == "true" ]]; then
            _amp_pfx="${AMPVALTEST_OUTPUT_PREFIX:-valtest_amplified}"
            _cue_skip_test="${_cue_skip_test} && test -s '${out_cue}/${_amp_pfx}_metric_summary_test.json' && test -s '${out_cue}/${_amp_pfx}_predictions_test.csv' && test -s '${out_cue}/${_amp_pfx}_predictions_validation.csv'"
          fi
          _cue_skip_ok=0
          if [[ "${SKIP_IF_OUTPUT_EXISTS}" == "1" ]]; then
            # Skip when complete cue outputs already exist on HPC (file presence only).
            # Do NOT call check_mech_checkpoint on login SSH (no reliable python there).
            if _hpc_ssh "${HPC_USER}@${HPC_HOST}" "${_cue_skip_test}"; then
              _cue_skip_ok=1
            fi
          fi
          if [[ "${_cue_skip_ok}" == "1" ]]; then
            echo "SKIP CUE (complete outputs exist on HPC): ${out_cue}"
            SUBMIT_SBATCH_SKIPPED_EXISTING=$((SUBMIT_SBATCH_SKIPPED_EXISTING + 1))
            continue
          fi
          node="$(_pick_sbatch_nodelist)"
          sb_common="--partition=${SBATCH_PARTITION} --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=${SBATCH_CPUS} --mem=${SBATCH_MEM} --time=${SBATCH_TIME} --nodelist=${node}"
          _init_bb_cue=""
          if ! _init_bb_cue="$(_warmft_sbatch_export_frag "${run_root}")"; then
            if [[ "${CHECK_INPUTS_STRICT}" == "1" ]]; then
              exit 1
            fi
            SUBMIT_SKIPPED_MISSING_INPUTS=$((SUBMIT_SKIPPED_MISSING_INPUTS + 1))
            continue
          fi
          # TRAIN_ON_TRAIN_VAL=1: train cue on train∪val (hard cues from both); no val early-stop / τ retune.
          # FIXED_THRESHOLD / TUNE_SPLIT optional; slurm defaults τ=0.5 when TRAIN_ON_TRAIN_VAL=1.
          _tv_exp=""
          if [[ "${TRAIN_ON_TRAIN_VAL:-0}" == "1" ]]; then
            _tv_exp=",TRAIN_ON_TRAIN_VAL=1"
            [[ -n "${FIXED_THRESHOLD:-}" ]] && _tv_exp="${_tv_exp},FIXED_THRESHOLD=${FIXED_THRESHOLD}"
            [[ -n "${TUNE_SPLIT:-}" ]] && _tv_exp="${_tv_exp},TUNE_SPLIT=${TUNE_SPLIT}"
          fi
          _ampvt_exp=""
          if [[ "${AMPVALTEST:-0}" == "1" ]]; then
            if [[ -z "${LLM_CSV_VALIDATION:-}" || -z "${LLM_CSV_TEST:-}" ]]; then
              echo "ERROR: AMPVALTEST=1 requires LLM_CSV_VALIDATION and LLM_CSV_TEST" >&2
              exit 1
            fi
            _ampvt_exp=",AMPVALTEST=1,LLM_CSV_VALIDATION=${LLM_CSV_VALIDATION},LLM_CSV_TEST=${LLM_CSV_TEST}"
            _ampvt_exp="${_ampvt_exp},RESELECT_BEST_EPOCH=${RESELECT_BEST_EPOCH:-1}"
            _ampvt_exp="${_ampvt_exp},AMPVALTEST_OUTPUT_PREFIX=${AMPVALTEST_OUTPUT_PREFIX:-valtest_amplified}"
          fi
          _scale_exp=",E_SCALE_MODE=${E_SCALE_MODE},E_INIT=${e_init}"
          if [[ "${E_SCALE_MODE}" == "split" ]]; then
            _scale_exp="${_scale_exp},A_INIT=${a_init},T_INIT=${t_init}"
          fi
          _scale_exp="${_scale_exp},DUAL_MIX_INIT=${DUAL_MIX_INIT:--1.0},DUAL_HARD_GATE_RAMP_RATIO=${DUAL_HARD_GATE_RAMP_RATIO:-0.0},DUAL_MECH_PROJ_INIT_STD=${DUAL_MECH_PROJ_INIT_STD:-0.02}"
          _span_exp=",SPAN_OP=${SPAN_OP:-amplify},SPAN_CHANNELS=${SPAN_CHANNELS:-both},LLM_NORMAL_COL=${LLM_NORMAL_COL:-normal_spans}"
          case "${SPAN_OP:-amplify}" in
            deamplify|deamp|subtract|dampen|attn_deamp|attn_deamplify|key_damp|attention_deamp)
              _span_exp="${_span_exp},DEAMP_AUTO=${DEAMP_AUTO:-1},DEAMP_TEACHER=${DEAMP_TEACHER:-}"
              if [[ -n "${SPAN_COVERAGE_MAX:-}" ]]; then
                _span_exp="${_span_exp},SPAN_COVERAGE_MAX=${SPAN_COVERAGE_MAX},DEAMP_AUTO=0"
              fi
              if [[ -n "${ATTN_DEAMP_LAYERS:-}" ]]; then
                _span_exp="${_span_exp},ATTN_DEAMP_LAYERS=${ATTN_DEAMP_LAYERS},DEAMP_AUTO=0"
              fi
              if [[ -n "${LEFTOVER_MASK_DROPOUT:-}" ]]; then
                _span_exp="${_span_exp},LEFTOVER_MASK_DROPOUT=${LEFTOVER_MASK_DROPOUT},DEAMP_AUTO=0"
              fi
              ;;
          esac
          exp_cue="ALL,CONDA_ENV=${CONDA_ENV},ECHOHATE_SPLITTED_DATA_SUBDIR=${split_sub},DATA_ROOT=${ENCODER_DATA_ROOT},DATASET=${ds},MODEL_NAME=${MODEL_NAME},POS_WEIGHT_MODE=${posw},LLM_CSV=${llm_hpc},OUTPUT_DIR=${out_cue},NUM_EPOCHS=${NUM_EPOCHS},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS},GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING},LEARNING_RATE=${LEARNING_RATE},WEIGHT_DECAY=${WEIGHT_DECAY},WARMUP_RATIO=${WARMUP_RATIO},MAX_LENGTH=${MAX_LENGTH},RANDOM_STATE=${RANDOM_STATE},CE_CONTRASTIVE_LAMBDA=${CE_CONTRASTIVE_LAMBDA},CONTRASTIVE_MARGIN=${CONTRASTIVE_MARGIN}${_scale_exp},AMP_POOL=${AMP_POOL:-sum},FUSION_LEVEL=${FUSION_LEVEL:-post},LAMBDA_MECH=${cue_lambda_mech},LAMBDA_MECH_RAMP_RATIO=${CUE_LAMBDA_MECH_RAMP_RATIO},CUE_MODE=${CUE_MODE},EVALUATION_SPLIT=validation,RUN_EVAL_OVERWRITE_OUTPUTS=${RUN_EVAL_OVERWRITE_OUTPUTS},FORCE_RETRAIN=${FORCE_RETRAIN},TRACK_DYNAMICS=${TRACK_DYNAMICS},HATE_MECH_TAXONOMY=${HATE_MECH_TAXONOMY},ALLOW_TAXONOMY_OVERWRITE=${ALLOW_TAXONOMY_OVERWRITE:-0}${_tv_exp}${_ampvt_exp}${_init_bb_cue}${_span_exp}"
          jc="${TEACHER_PATH_SLUG}_fft_cue_${_E}_${JOB_DS_TAG}_${mode}_${scale_tag}_lm${clm_tag}${STUDENT_OUT_SUFFIX}"
          if [[ "${E_SCALE_MODE}" == "split" ]]; then
            echo "Submit CUE ${JOB_DS_TAG} ${mode} a=${a_init} t=${t_init} cue_lambda_mech=${cue_lambda_mech} hate_mech=${HATE_MECH_TAXONOMY} SPAN_OP=${SPAN_OP:-amplify} SPAN_CHANNELS=${SPAN_CHANNELS:-both} CL=${CE_CONTRASTIVE_LAMBDA} -> nodelist=${node}  out=${out_cue}"
          else
            echo "Submit CUE ${JOB_DS_TAG} ${mode} e_init=${e_init} cue_lambda_mech=${cue_lambda_mech} hate_mech=${HATE_MECH_TAXONOMY} SPAN_OP=${SPAN_OP:-amplify} SPAN_CHANNELS=${SPAN_CHANNELS:-both} CL=${CE_CONTRASTIVE_LAMBDA} -> nodelist=${node}  out=${out_cue}"
          fi
          hpc_ssh_sbatch_export_all "${HPC_DIR}" "${SLURM_CUE}" "${exp_cue}" "--job-name=${jc} --output=${jc}_%j.out --error=${jc}_%j.err ${sb_common}" "unset CUDA_VISIBLE_DEVICES" || exit 1
          SUBMIT_SBATCH_OK=$((SUBMIT_SBATCH_OK + 1))
          echo "  JOB_ID=${HPC_LAST_SSH_JOB_ID}  CL=${CE_CONTRASTIVE_LAMBDA}"
        done
      done
    fi
  done
done
done
done

echo "=========================================="
echo "Submit summary: sbatch successes=${SUBMIT_SBATCH_OK}  skipped_existing_output=${SUBMIT_SBATCH_SKIPPED_EXISTING}  skipped_missing_inputs=${SUBMIT_SKIPPED_MISSING_INPUTS}"
echo "Done. squeue: ssh ${HPC_USER}@${HPC_HOST} 'squeue -u ${HPC_USER}'"
