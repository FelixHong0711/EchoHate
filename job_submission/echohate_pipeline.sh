#!/usr/bin/env bash
# EchoHate pipeline entry point (job_submission layout).
#
#   bash job_submission/echohate_pipeline.sh help
#   bash job_submission/echohate_pipeline.sh preprocessing split --datasets IHC
#   bash job_submission/echohate_pipeline.sh stage1 select_hard --method var25 --seed-dir seed_0_mech --mirror-selection-gmm
#   bash job_submission/echohate_pipeline.sh stage2 roberta_baseline submit
#   bash job_submission/echohate_pipeline.sh stage2 cue_mech_attnmask submit

set -euo pipefail

_JS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ECHOHATE_ROOT="${ECHOHATE_ROOT:-$(cd "${_JS}/.." && pwd)}"

usage() {
  cat <<'EOF'
EchoHate job_submission layout

  preprocessing/          python -m pipeline.preprocessing (7 datasets, 811 splits)
  stage1/
    select_hard / var25 / rand   hard-row selection
  stage2/
    roberta_baseline/       standalone RoBERTa FFT
    cue_mech_attnmask/      mechanism-supervised encoder submit

LLM teacher inference and uncertainty-guided deferral are not in this release.
They will be added when the full code is released.

Examples:
  bash job_submission/echohate_pipeline.sh preprocessing run split --datasets DynaHate
  bash job_submission/echohate_pipeline.sh stage1 select_hard --method var25 --seed-dir seed_0_mech --mirror-selection-gmm
  bash job_submission/echohate_pipeline.sh stage2 roberta_baseline submit_roberta_baseline_from_local.sh

See README.md. Python: python -m pipeline help
EOF
}

_stage="${1:-help}"
shift || true

case "${_stage}" in
  help|-h|--help)
    usage
    exit 0
    ;;
  preprocessing)
    exec bash "${_JS}/preprocessing/run.sh" "$@"
    ;;
  stage1)
    _s1="${1:-help}"
    shift || true
    case "${_s1}" in
      select_hard|hard|var25|rand|rand50|rand25)
        if [[ "${_s1}" == var25 || "${_s1}" == rand25 || "${_s1}" == rand50 || "${_s1}" == rand ]]; then
          _method="${_s1}"
          [[ "${_method}" == rand ]] && _method=rand50
          exec python -m pipeline.stage1.select_hard_cases_for_mech --method "${_method}" "$@"
        fi
        exec python -m pipeline.stage1.select_hard_cases_for_mech "$@"
        ;;
      llm_infer_on_hard|infer_hard|selected_hard|llm_validation_test|validation_test|llm_inference|gmm|selection_gmm|select|toy_prompt_infer|toy_prompt)
        echo "ERROR: LLM teacher inference is not in this release. It will be included with the full code upon acceptance." >&2
        exit 1
        ;;
      *)
        echo "stage1 subcommand: select_hard" >&2
        exit 1
        ;;
    esac
    ;;
  stage2)
    _s2="${1:-help}"
    shift || true
    case "${_s2}" in
      roberta_baseline|baseline)
        _script="${1:-submit_roberta_baseline_from_local.sh}"
        shift || true
        exec bash "${_JS}/stage2/roberta_baseline/${_script}" "$@"
        ;;
      cue_mech_attnmask|cue|mech|attnmask)
        _script="${1:-submit_roberta_fft_mech_cue_hpclogin_dgx_from_local.sh}"
        shift || true
        if [[ "${_script}" == submit || "${_script}" == submit_post_llm ]]; then
          _script="submit_roberta_fft_mech_cue_hpclogin_dgx_from_local.sh"
        fi
        if [[ "${_script}" == slurm_cue_distill.sh ]]; then
          echo "ERROR: use submit_roberta_fft_mech_cue_hpclogin_dgx_from_local.sh (APPROACH=mech)." >&2
          exit 1
        fi
        if [[ -z "${_script}" ]]; then
          echo "stage2 cue_mech_attnmask: pass submit/slurm script name" >&2
          ls "${_JS}/stage2/cue_mech_attnmask/" >&2
          exit 1
        fi
        exec bash "${_JS}/stage2/cue_mech_attnmask/${_script}" "$@"
        ;;
      analyze_infer|hybrid|lambda)
        echo "ERROR: uncertainty-guided LLM deferral is not in this release. It will be included with the full code upon acceptance." >&2
        exit 1
        ;;
      *)
        echo "stage2 subcommand: roberta_baseline | cue_mech_attnmask" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unknown stage: ${_stage}" >&2
    usage >&2
    exit 1
    ;;
esac
