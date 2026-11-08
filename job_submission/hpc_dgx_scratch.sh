#!/usr/bin/env bash
# =============================================================================
# Deakin DGX local NVMe + cluster /scratch staging for Slurm jobs.
#
# DGX nodes: Slurm creates /raid/$SLURM_JOB_ID and deletes it when the job ends.
# Cluster /scratch: fallback when /raid is missing (CPU partitions, login probes).
#
# High-I/O (HF caches, tmp, checkpoints, predictions) must NOT run from $HOME NFS.
# Stage inputs onto /raid, write outputs there, copy results back to the durable
# path (usually /scratch/$USER/RA — not home).
#
# Sourced from hpc_env_deakin.sh. Safe to source more than once.
# Opt out: HPC_SCRATCH_DISABLE=1
# Skip seeding ~/.cache/huggingface: HPC_SCRATCH_SEED_HF=0
# =============================================================================

_HPC_SCRATCH_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

hpc_scratch_is_home_path() {
  case "${1:-}" in
    /home/*|/Users/*) return 0 ;;
    *) return 1 ;;
  esac
}

# Paths that should be staged onto /raid for ML I/O (Julian DGX policy).
# Includes cluster /scratch durable trees; excludes /scratch/$USER/jobs/$JOB_ID
# (already the fallback job scratch).
hpc_scratch_is_stageable_path() {
  case "${1:-}" in
    /home/*|/Users/*) return 0 ;;
    /scratch/*/jobs/*) return 1 ;;
    /scratch/*) return 0 ;;
    *) return 1 ;;
  esac
}

hpc_scratch_abs() {
  local p="${1:-}"
  if [ -z "${p}" ]; then
    return 0
  fi
  case "${p}" in
    /*) printf '%s' "${p}" ;;
    *) printf '%s' "$(pwd)/${p}" ;;
  esac
}

_hpc_scratch_copy() {
  local src="$1"
  local dest="$2"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${src}" "${dest}"
  else
    mkdir -p "$(dirname "${dest}")"
    cp -a "${src}" "${dest}"
  fi
}

hpc_scratch_register_stage_out() {
  local src="${1:-}"
  local dest="${2:-}"
  [ -n "${src}" ] && [ -n "${dest}" ] || return 0
  [ -n "${HPC_SCRATCH:-}" ] || return 0
  if [ "${src}" = "${dest}" ]; then
    return 0
  fi
  mkdir -p "${HPC_SCRATCH}"
  local map="${HPC_SCRATCH}/.stage_out.tsv"
  if [ -f "${map}" ] && grep -Fqx "${src}	${dest}" "${map}" 2>/dev/null; then
    return 0
  fi
  printf '%s\t%s\n' "${src}" "${dest}" >> "${map}"
}

hpc_scratch_purge_heavy_artifacts() {
  # Julian: physically delete HF checkpoints / optimizer shards on /raid before
  # stage-out. Do NOT rely on grepping .stage_out.tsv folder names.
  local root="${1:-${HPC_SCRATCH:-}}"
  [ -n "${root}" ] && [ -d "${root}" ] || return 0
  echo "[scratch] purging checkpoint-* / trainer weight dumps under ${root}"
  # Depth-first: -prune + rm so find does not walk into deleted trees.
  find "${root}" -type d \( \
      -name 'checkpoint-*' -o \
      -name 'checkpoints' -o \
      -name 'checkpoint_epoch_*' \
    \) -prune -exec rm -rf {} + 2>/dev/null || true
  # Leftover optimizer / RNG / scheduler shards (keep final lightweight JSON/metrics).
  find "${root}" -type f \( \
      -name 'optimizer.pt' -o \
      -name 'scheduler.pt' -o \
      -name 'rng_state.pth' -o \
      -name 'trainer_state.json' -o \
      -name 'training_args.bin' \
    \) -delete 2>/dev/null || true
}

hpc_scratch_finalize() {
  local rc=$?
  set +e
  if [ -z "${HPC_SCRATCH:-}" ] || [ ! -d "${HPC_SCRATCH}" ]; then
    return "${rc}"
  fi
  # Always purge before any sync (even if map empty — keeps /raid tidy if trap races).
  if [ "${HPC_SCRATCH_SKIP_CHECKPOINT_PURGE:-0}" != "1" ]; then
    hpc_scratch_purge_heavy_artifacts "${HPC_SCRATCH}"
  fi
  local map="${HPC_SCRATCH}/.stage_out.tsv"
  if [ ! -s "${map}" ]; then
    echo "[scratch] no stage-out map; /raid job dir is deleted at job end"
    return "${rc}"
  fi
  echo "[scratch] STAGE-OUT: copying results back to durable path ( /raid is deleted when this job ends)"
  local src dest ok=1
  # Exclude HF epoch dumps even if a path still somehow remains.
  local _rsync_ex=(
    --exclude='checkpoint-*/'
    --exclude='checkpoints/'
    --exclude='checkpoint_epoch_*/'
    --exclude='optimizer.pt'
    --exclude='scheduler.pt'
    --exclude='rng_state.pth'
    --exclude='global_step*/'
  )
  while IFS=$'\t' read -r src dest; do
    [ -n "${src}" ] && [ -n "${dest}" ] || continue
    if [ ! -e "${src}" ]; then
      echo "[scratch] WARN: stage-out source missing: ${src}" >&2
      continue
    fi
    echo "[scratch]   ${src}  ->  ${dest}"
    mkdir -p "$(dirname "${dest}")"
    if [ -d "${src}" ]; then
      mkdir -p "${dest}"
      if command -v rsync >/dev/null 2>&1; then
        if ! rsync -a "${_rsync_ex[@]}" "${src}/" "${dest}/"; then
          echo "[scratch] ERROR: rsync failed: ${src} -> ${dest}" >&2
          ok=0
        fi
      else
        if ! cp -a "${src}/." "${dest}/"; then
          echo "[scratch] ERROR: cp failed: ${src} -> ${dest}" >&2
          ok=0
        fi
        # Best-effort purge of any checkpoint dirs that slipped through cp -a.
        find "${dest}" -type d \( -name 'checkpoint-*' -o -name 'checkpoints' -o -name 'checkpoint_epoch_*' \) \
          -prune -exec rm -rf {} + 2>/dev/null || true
      fi
    else
      if ! _hpc_scratch_copy "${src}" "${dest}"; then
        echo "[scratch] ERROR: copy failed: ${src} -> ${dest}" >&2
        ok=0
      fi
    fi
  done < "${map}"
  if [ "${ok}" != "1" ]; then
    echo "[scratch] ERROR: one or more stage-out copies failed; data on /raid will be deleted" >&2
    if [ "${rc}" -eq 0 ]; then
      rc=1
    fi
  else
    echo "[scratch] stage-out complete"
  fi
  return "${rc}"
}

hpc_scratch_init() {
  if [ "${HPC_SCRATCH_INIT_DONE:-0}" = "1" ]; then
    return 0
  fi
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    return 0
  fi
  if [ "${HPC_SCRATCH_DISABLE:-0}" = "1" ]; then
    echo "[scratch] disabled (HPC_SCRATCH_DISABLE=1); I/O stays on home"
    HPC_SCRATCH_INIT_DONE=1
    return 0
  fi

  local scratch=""
  if [ -d "/raid/${SLURM_JOB_ID}" ]; then
    scratch="/raid/${SLURM_JOB_ID}"
  elif [ -d "/raid" ] && [ -w "/raid" ]; then
    scratch="/raid/${SLURM_JOB_ID}"
    mkdir -p "${scratch}" || scratch=""
  fi
  if [ -z "${scratch}" ] && [ -d "/scratch" ] && [ -w "/scratch" ]; then
    scratch="/scratch/${USER:-${LOGNAME:-hohuu}}/jobs/${SLURM_JOB_ID}"
    mkdir -p "${scratch}" || scratch=""
    if [ -n "${scratch}" ]; then
      echo "[scratch] /raid/\$SLURM_JOB_ID missing; using cluster /scratch (temporary, not backed up)"
    fi
  fi
  if [ -z "${scratch}" ]; then
    echo "[scratch] WARN: neither /raid/\$SLURM_JOB_ID nor /scratch is usable; I/O stays on home" >&2
    HPC_SCRATCH_INIT_DONE=1
    return 0
  fi

  export HPC_SCRATCH="${scratch}"
  mkdir -p "${HPC_SCRATCH}/tmp" "${HPC_SCRATCH}/hf" "${HPC_SCRATCH}/torch" \
    "${HPC_SCRATCH}/xdg" "${HPC_SCRATCH}/stage" "${HPC_SCRATCH}/out" \
    "${HPC_SCRATCH}/work" "${HPC_SCRATCH}/wandb" "${HPC_SCRATCH}/mpl" \
    "${HPC_SCRATCH}/triton"
  export TMPDIR="${HPC_SCRATCH}/tmp"
  export TMP="${TMPDIR}"
  export TEMP="${TMPDIR}"
  export HF_HOME="${HPC_SCRATCH}/hf"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export TORCH_HOME="${HPC_SCRATCH}/torch"
  export XDG_CACHE_HOME="${HPC_SCRATCH}/xdg"
  export WANDB_DIR="${HPC_SCRATCH}/wandb"
  export WANDB_CACHE_DIR="${HPC_SCRATCH}/wandb"
  export MPLCONFIGDIR="${HPC_SCRATCH}/mpl"
  export TRITON_CACHE_DIR="${HPC_SCRATCH}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${HPC_SCRATCH}/torchinductor"
  : > "${HPC_SCRATCH}/.stage_out.tsv"

  echo "[scratch] HPC_SCRATCH=${HPC_SCRATCH}"
  echo "[scratch] TMPDIR=${TMPDIR} HF_HOME=${HF_HOME}"
  if command -v df >/dev/null 2>&1; then
    df -h "${HPC_SCRATCH}" 2>/dev/null | tail -1 | awk '{print "[scratch] disk: "$0}'
  fi

  if [ "${HPC_SCRATCH_SEED_HF:-1}" = "1" ]; then
    local _src="${HOME}/.cache/huggingface"
    if [ -d "${_src}" ]; then
      local _kb
      _kb="$(du -sk "${_src}" 2>/dev/null | awk '{print $1}')"
      local _max_kb=$(( ${HPC_SCRATCH_MAX_HF_SEED_GB:-40} * 1024 * 1024 ))
      if [ -n "${_kb}" ] && [ "${_kb}" -gt "${_max_kb}" ]; then
        echo "[scratch] skip HF seed: ${_src} is ${_kb} KiB (>${HPC_SCRATCH_MAX_HF_SEED_GB:-40}GB)"
      else
        echo "[scratch] seeding HF cache from ${_src} (avoids re-download on NVMe)"
        mkdir -p "${HF_HOME}"
        if command -v rsync >/dev/null 2>&1; then
          rsync -a "${_src}/" "${HF_HOME}/" || echo "[scratch] WARN: HF cache seed rsync failed" >&2
        else
          cp -a "${_src}/." "${HF_HOME}/" || true
        fi
      fi
    fi
  fi

  if [ -z "${HPC_SCRATCH_TRAP_INSTALLED:-}" ]; then
    # Copy results back even on failure; /raid is wiped when the job ends.
    trap hpc_scratch_finalize EXIT
    HPC_SCRATCH_TRAP_INSTALLED=1
  fi
  HPC_SCRATCH_INIT_DONE=1
}

hpc_scratch_rel_under_user() {
  # /home/remote/hohuu/RA/foo → RA/foo
  # /scratch/hohuu/RA/foo → RA/foo
  # /Users/x/RA/foo → RA/foo
  local p="$1"
  if [[ "${p}" =~ ^/home/remote/[^/]+/(.*)$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  elif [[ "${p}" =~ ^/home/[^/]+/(.*)$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  elif [[ "${p}" =~ ^/scratch/[^/]+/(.*)$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  elif [[ "${p}" =~ ^/Users/[^/]+/(.*)$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    printf '%s' "${p#/}"
  fi
}

hpc_scratch_bind_output() {
  local var="$1"
  [ -n "${HPC_SCRATCH:-}" ] || return 0
  local home_path
  home_path="$(hpc_scratch_abs "${!var:-}")"
  [ -n "${home_path}" ] || return 0
  case "${home_path}" in
    "${HPC_SCRATCH}"/*) return 0 ;;
  esac
  hpc_scratch_is_stageable_path "${home_path}" || return 0

  local rel scratch_path as_file=0
  rel="$(hpc_scratch_rel_under_user "${home_path}")"
  scratch_path="${HPC_SCRATCH}/out/${rel}"
  case "${home_path}" in
    *.csv|*.json|*.jsonl|*.pt|*.bin|*.tsv|*.txt|*.pkl|*.pth|*.safetensors) as_file=1 ;;
  esac
  if [ -f "${home_path}" ]; then
    as_file=1
  elif [ -d "${home_path}" ]; then
    as_file=0
  fi
  if [ "${as_file}" = "1" ]; then
    mkdir -p "$(dirname "${scratch_path}")"
    if [ -f "${home_path}" ]; then
      echo "[scratch] stage-in existing ${var}: ${home_path}"
      _hpc_scratch_copy "${home_path}" "${scratch_path}"
    fi
  else
    mkdir -p "${scratch_path}"
    if [ -d "${home_path}" ]; then
      echo "[scratch] stage-in existing ${var}: ${home_path}"
      if command -v rsync >/dev/null 2>&1; then
        rsync -a "${home_path}/" "${scratch_path}/" || true
      else
        cp -a "${home_path}/." "${scratch_path}/" || true
      fi
    fi
  fi
  printf -v "${var}" '%s' "${scratch_path}"
  export "${var}"
  hpc_scratch_register_stage_out "${scratch_path}" "${home_path}"
  echo "[scratch] ${var}: ${home_path}  →  ${scratch_path} (stage-out to durable path on EXIT)"
}

hpc_scratch_stage_in() {
  local var="$1"
  [ -n "${HPC_SCRATCH:-}" ] || return 0
  local src
  src="$(hpc_scratch_abs "${!var:-}")"
  [ -n "${src}" ] || return 0
  case "${src}" in
    "${HPC_SCRATCH}"/*) return 0 ;;
  esac
  hpc_scratch_is_stageable_path "${src}" || return 0
  [ -e "${src}" ] || return 0

  local rel dest
  rel="$(hpc_scratch_rel_under_user "${src}")"
  dest="${HPC_SCRATCH}/stage/${rel}"
  mkdir -p "$(dirname "${dest}")"
  echo "[scratch] stage-in ${var}: ${src}  →  ${dest}"
  if [ -d "${src}" ]; then
    mkdir -p "${dest}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "${src}/" "${dest}/"
    else
      cp -a "${src}/." "${dest}/"
    fi
  else
    _hpc_scratch_copy "${src}" "${dest}"
  fi
  printf -v "${var}" '%s' "${dest}"
  export "${var}"
}

# Copy a project tree onto scratch and register named subdirs for stage-out.
# Usage: hpc_scratch_mirror_project SRC DEST_NAME [subdir ...]
# Prints the scratch path. Typical subdirs: save clustered_dataset preprocessed_data
hpc_scratch_mirror_project() {
  local src="${1:-}"
  local name="${2:-}"
  shift 2 || true
  [ -n "${HPC_SCRATCH:-}" ] || { printf '%s' "${src}"; return 0; }
  [ -n "${src}" ] && [ -n "${name}" ] || return 1
  src="$(hpc_scratch_abs "${src}")"
  local dest="${HPC_SCRATCH}/work/${name}"
  mkdir -p "${dest}"
  echo "[scratch] STAGE-IN project ${src} -> ${dest}" >&2
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude '.git/' \
      --exclude 'logs/' \
      --exclude '__pycache__/' \
      --exclude '.conda/' \
      --exclude 'node_modules/' \
      "${src}/" "${dest}/"
  else
    cp -a "${src}/." "${dest}/"
  fi
  local sub
  for sub in "$@"; do
    [ -n "${sub}" ] || continue
    mkdir -p "${dest}/${sub}"
    hpc_scratch_register_stage_out "${dest}/${sub}" "${src}/${sub}"
  done
  printf '%s' "${dest}"
}

hpc_scratch_auto_bind() {
  [ -n "${HPC_SCRATCH:-}" ] || return 0
  if [ "${HPC_SCRATCH_DISABLE:-0}" = "1" ]; then
    return 0
  fi

  local var
  for var in OUTPUT_DIR SAVE_ROOT AMPLEHATE_INFER_SAVE_ROOT RVHATE_DUMP_OUT \
    AMPLEHATE_SAVE_DIR SHAREDCON_OUT_DIR LAHN_OUT_DIR ENCODER_RUN_DIR; do
    if [ -n "${!var:-}" ]; then
      hpc_scratch_bind_output "${var}"
    fi
  done

  for var in INPUT_CSV LLM_CSV SEEDS_CSV CONTEXT_PATH STANCE_TAXONOMY_JSON \
    AMPLEHATE_LLM_MECH_CSV ECHOHATE_SPLIT_DIR INIT_BACKBONE_FROM; do
    if [ -n "${!var:-}" ]; then
      hpc_scratch_stage_in "${var}"
    fi
  done
}
