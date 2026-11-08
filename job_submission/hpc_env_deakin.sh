#!/usr/bin/env bash
# =============================================================================
# Deakin Research HPC — hpc.it.deakin.edu.au (Ubuntu 22.04, Lmod, partition "gpu")
# Also usable on other Linux hosts with Miniconda/Anaconda: set HPC_LOAD_CUDA_MODULE=0 to skip Lmod CUDA.
# =============================================================================
# Source this file from Slurm scripts *after* `cd "${SLURM_SUBMIT_DIR}"` (EchoHate repo root).
#
# Optional env (via sbatch --export=...):
#   CONDA_ENV         default: rag_v8_gpu
#   CREATE_ENV        default: 0  (set 1 once to create env)
#   HPC_MODULE_CUDA   default: CUDA/12.2.0 on hpclogin DGX (EasyBuild); cuda/12.2 on older hpc2-style stacks
#   HPC_MODULE_CUDA_FALLBACKS  space-separated extra modules to try (e.g. "CUDA/12.6.0 cuda/12.1")
#   HPC_LOAD_CUDA_MODULE  set 1 to Lmod-load CUDA on login node (default: load only in Slurm jobs)
#   HPC_MODULE_PURGE      set 1 to run `module purge` in Slurm jobs (default 0 — purge breaks some DGX stacks)
#   HPC_REQUIRE_CUDA      set 1 in Slurm jobs to exit if torch.cuda.is_available() is False (default 1)
#   HPC_SCRATCH_DISABLE   set 1 to keep I/O on durable FS (default 0: /raid/$SLURM_JOB_ID then stage-out to /scratch/$USER/RA)
#   HPC_RA                durable project root — prefer /scratch/$USER/RA, never home for ML I/O
#
# Requires: Miniconda or Anaconda. Search order:
#   $CONDA_BASE (if set), /opt/miniconda3, $HOME/miniconda3, $HOME/anaconda3
# Use python -m pip so packages land in the active env (avoids ~/.local when pip is wrong).
# =============================================================================

_HPC_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  # shellcheck source=/dev/null
  source "${_HPC_ENV_DIR}/hpc_cuda_slurm_defaults.sh"
  # DGX local NVMe (/raid/$SLURM_JOB_ID) or cluster /scratch — not $HOME.
  # shellcheck source=/dev/null
  source "${_HPC_ENV_DIR}/hpc_dgx_scratch.sh"
  hpc_scratch_init
fi

# In Slurm GPU jobs, `module purge` often removes the scheduler/CUDA stack (then torch sees cuda: False).
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ "${HPC_MODULE_PURGE:-0}" = "1" ]; then
    module purge 2>/dev/null || true
  fi
else
  module purge 2>/dev/null || true
fi

_hpc_prepend_ld_for_torch() {
  local _d _pyver _lc _dd
  if [ -n "${CONDA_PREFIX:-}" ]; then
    _pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3.10")"
    for _d in \
      "${CONDA_PREFIX}/lib/python${_pyver}/site-packages/torch/lib" \
      "${CONDA_PREFIX}/lib/python${_pyver}/site-packages/nvidia/cuda_runtime/lib" \
      "${CONDA_PREFIX}/lib"; do
      if [ -d "${_d}" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
          *":${_d}:"*) ;;
          *) export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}" ;;
        esac
      fi
    done
  fi
  if command -v ldconfig >/dev/null 2>&1; then
    _lc="$(ldconfig -p 2>/dev/null | grep 'libcuda\.so' | awk '{print $NF; exit}')"
    if [ -n "${_lc}" ]; then
      _dd="$(dirname "$(readlink -f "${_lc}" 2>/dev/null || echo "${_lc}")")"
      if [ -n "${_dd}" ] && [ -d "${_dd}" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
          *":${_dd}:"*) ;;
          *) export LD_LIBRARY_PATH="${_dd}:${LD_LIBRARY_PATH:-}" ;;
        esac
      fi
    fi
  fi
  for _d in /usr/lib/x86_64-linux-gnu /usr/lib64/nvidia /usr/local/cuda/lib64; do
    if [ -d "${_d}" ]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":${_d}:"*) ;;
        *) export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}" ;;
      esac
    fi
  done
}

_hpc_module_load_one() {
  local _mod="$1"
  [ -z "${_mod}" ] && return 1
  if module load "${_mod}" 2>/dev/null; then
    return 0
  fi
  if module --ignore_cache load "${_mod}" 2>/dev/null; then
    echo "[hpc_env_deakin] Loaded module (--ignore_cache): ${_mod}"
    return 0
  fi
  return 1
}

_hpc_try_load_cuda_module() {
  local _mod _loaded="" _candidates=()
  if [ -n "${HPC_MODULE_CUDA:-}" ]; then
    _candidates+=("${HPC_MODULE_CUDA}")
  fi
  if [ -n "${HPC_MODULE_CUDA_FALLBACKS:-}" ]; then
    # '+' from sbatch --export (spaces are unsafe there); also accept plain spaces.
    _fb="${HPC_MODULE_CUDA_FALLBACKS//+/ }"
    # shellcheck disable=SC2206
    _candidates+=(${_fb})
  fi
  # hpclogin DGX EasyBuild (CUDA/12.x) — try site default (D) and 12.2.2 before 12.2.0; then hpc2-style cuda/12.2.
  _candidates+=(
    "CUDA/12.6.0" "CUDA/12.2.2" "CUDA/12.2.0" "CUDA/11.8.0" "CUDA/11.7.0"
    "cuda/12.2" "cuda/12.1" "cuda/12.0" "cuda/11.8" "cuda/12"
  )
  for _mod in "${_candidates[@]}"; do
    [ -z "${_mod}" ] && continue
    if _hpc_module_load_one "${_mod}"; then
      echo "[hpc_env_deakin] Loaded module: ${_mod}"
      _loaded="${_mod}"
      return 0
    fi
  done
  if command -v module >/dev/null 2>&1; then
    while IFS= read -r _mod; do
      [ -z "${_mod}" ] && continue
      _mod="${_mod%%(*}"
      _mod="$(printf '%s' "${_mod}" | sed 's/[[:space:]]*$//')"
      if _hpc_module_load_one "${_mod}"; then
        echo "[hpc_env_deakin] Loaded module (from avail): ${_mod}"
        _loaded="${_mod}"
        return 0
      fi
    done < <(module -t avail cuda 2>/dev/null | awk '{print $1}' | grep -iE '^(cuda|CUDA)/' | head -12)
  fi
  if [ -z "${SLURM_JOB_ID:-}" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[hpc_env_deakin] NOTE: no CUDA module on this host (likely login node). GPU jobs load modules on DGX compute nodes."
  else
    echo "[hpc_env_deakin] WARNING: could not load any CUDA module (tried: ${_candidates[*]})."
    echo "[hpc_env_deakin]         On a DGX node: module avail cuda  then export HPC_MODULE_CUDA=<exact name>"
  fi
  return 1
}

# CUDA module: prefer loading inside GPU jobs only (login node often has no GPU; avoids noisy checks).
_load_cuda_mod=0
if [ -n "${SLURM_JOB_ID:-}" ]; then
  _load_cuda_mod=1
elif [ "${HPC_LOAD_CUDA_MODULE:-0}" = "1" ]; then
  _load_cuda_mod=1
fi

if [ "${_load_cuda_mod}" = "1" ] && [ "${HPC_LOAD_CUDA_MODULE:-1}" != "0" ]; then
  if [ -n "${HPC_MODULE_CUDA+x}" ] && [ -z "${HPC_MODULE_CUDA}" ]; then
    : # HPC_MODULE_CUDA="" → skip Lmod load (use Slurm-provided driver only)
  else
    _hpc_try_load_cuda_module || true
  fi
fi

CONDA_ENV="${CONDA_ENV:-rag_v8_gpu}"
# Slurm --export=...,'name'... can leak literal quotes into the env name.
CONDA_ENV="${CONDA_ENV#\'}"; CONDA_ENV="${CONDA_ENV%\'}"
CONDA_ENV="${CONDA_ENV#\"}"; CONDA_ENV="${CONDA_ENV%\"}"
CREATE_ENV="${CREATE_ENV:-0}"
CREATE_ENV="${CREATE_ENV#\'}"; CREATE_ENV="${CREATE_ENV%\'}"

# Non-interactive Slurm jobs: never prompt; conda 24+ may require channel ToS before create.
export CONDA_ALWAYS_YES="${CONDA_ALWAYS_YES:-true}"

_CONDA_SH=""
if [ -n "${CONDA_BASE:-}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  _CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
  _CONDA_SH="/opt/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  _CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  _CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if [ -z "${_CONDA_SH}" ]; then
  echo "[hpc_env_deakin] ERROR: Could not find conda.sh (tried \$CONDA_BASE, /opt/miniconda3, ~/miniconda3, ~/anaconda3)."
  echo "  Set CONDA_BASE to your install prefix or install Miniconda (see README_HPC_DEAKIN.md)."
  exit 1
fi
# shellcheck source=/dev/null
source "${_CONDA_SH}"
echo "[hpc_env_deakin] Using conda from: ${_CONDA_SH}"

# Never install into ~/.local during this script (pip falls back when base env is not writeable).
export PIP_USER=0
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
mkdir -p "${HOME}/.conda/envs"

# Fast-path: put ~/.conda/envs/<env>/bin on PATH without `conda shell hook`,
# `conda config`, `conda env list`, or `conda activate`. Those hit home NFS locks and
# have hung for 45–60+ min on hpc-dgx-b-2 while sibling jobs on b-3 finished in ~14 min.
_hpc_conda_env_prefix() {
  local n="$1"
  local cand
  for cand in \
    "${HOME}/.conda/envs/${n}" \
    "${HOME}/miniconda3/envs/${n}" \
    "${HOME}/anaconda3/envs/${n}" \
    "/opt/miniconda3/envs/${n}"
  do
    if [ -x "${cand}/bin/python" ]; then
      printf '%s' "${cand}"
      return 0
    fi
  done
  # Slow fallback only when CREATE_ENV needs discovery (avoid on every Slurm start).
  if [ "${HPC_CONDA_ALLOW_SLOW_LOOKUP:-0}" = "1" ]; then
    conda env list 2>/dev/null | grep -v '^#' | grep -v '^[[:space:]]*$' | sed 's/^[[:space:]]*\*//g' | awk -v n="$n" '$1==n {print $NF; exit}'
  fi
}

_hpc_activate_and_fix_path() {
  local _p
  _p="$(_hpc_conda_env_prefix "${CONDA_ENV}")"
  if [ -z "${_p}" ]; then
    echo "[hpc_env_deakin] ERROR: conda env '${CONDA_ENV}' not found under ~/.conda/envs (or miniconda3/envs)."
    echo "  On the login node: ls ~/.conda/envs ; export CONDA_ENV=... or CREATE_ENV=1"
    exit 1
  fi
  export CONDA_PREFIX="${_p}"
  export CONDA_DEFAULT_ENV="${CONDA_ENV}"
  # Prepend env bin; do not rely on conda activate (NFS-safe).
  case ":${PATH}:" in
    *":${CONDA_PREFIX}/bin:"*) ;;
    *) export PATH="${CONDA_PREFIX}/bin:${PATH}" ;;
  esac
  hash -r 2>/dev/null || true
  echo "[hpc_env_deakin] Fast-path env via PATH: ${CONDA_PREFIX}"
}

# Optional legacy activate (off by default). Only needed for CREATE_ENV=1 create path.
if [ "${CREATE_ENV:-0}" = "1" ] || [ "${HPC_CONDA_USE_ACTIVATE:-0}" = "1" ]; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  # Avoid rewriting ~/.condarc on every job (NFS flock). Only prepend if missing.
  if ! grep -q "${HOME}/.conda/envs" "${HOME}/.condarc" 2>/dev/null; then
    conda config --prepend envs_dirs "${HOME}/.conda/envs" 2>/dev/null || true
  fi
fi

# QLoRA (VRAM_SAFE=1 on LLM LoRA jobs) needs bitsandbytes in the active conda env.
# CREATE_ENV=1 installs it on fresh envs; existing rag_v8_gpu on hpc2 often lacks it.
_hpc_ensure_bitsandbytes() {
  if [ "${HPC_ENSURE_BITSANDBYTES:-1}" = "0" ]; then
    return 0
  fi
  if python - <<'PY' 2>/dev/null; then
import bitsandbytes as bnb

def _ver_tuple(s):
    parts = []
    for piece in str(s).split(".")[:3]:
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)

assert _ver_tuple(getattr(bnb, "__version__", "0")) >= (0, 46, 1)
PY
    return 0
  fi
  echo "[hpc_env_deakin] bitsandbytes>=0.46.1 missing in ${CONDA_ENV}; installing (QLoRA / VRAM_SAFE)..."
  python -m pip install --upgrade pip
  python -m pip install "bitsandbytes>=0.46.1" || {
    echo "[hpc_env_deakin] ERROR: bitsandbytes install failed. On the login node run:" >&2
    echo "  conda activate ${CONDA_ENV} && python -m pip install -U 'bitsandbytes>=0.46.1'" >&2
    echo "  Or resubmit with VRAM_SAFE=0 (full bf16 LoRA; may OOM on 14B)." >&2
    return 1
  }
}

if [ "${CREATE_ENV}" = "1" ] && [ -z "$(_hpc_conda_env_prefix "${CONDA_ENV}")" ]; then
  echo "[hpc_env_deakin] Creating conda env: ${CONDA_ENV}"
  # Newer conda requires accepting Anaconda channel Terms of Service once (batch jobs hang otherwise).
  if conda tos --help >/dev/null 2>&1; then
    echo "[hpc_env_deakin] Accepting conda channel ToS (non-interactive)..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
  fi
  conda create -n "${CONDA_ENV}" python=3.10 -y
  _hpc_activate_and_fix_path
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  _py="$(command -v python 2>/dev/null || true)"
  if [ "${_py}" = "/opt/miniconda3/bin/python" ] || [ "${_py}" = "/opt/miniconda3/bin/python3" ]; then
    echo "[hpc_env_deakin] ERROR: still using base Python after activate — conda env may be in a read-only path."
    echo "  Remove broken env: conda env remove -n ${CONDA_ENV} -y"
    echo "  Re-submit with CREATE_ENV=1 (env will be created under ~/.conda/envs if prepended envs_dirs works)."
    exit 1
  fi
  python -m pip install --upgrade pip
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
  python -m pip install "accelerate>=1.1.0"
  python -m pip install "bitsandbytes>=0.46.1"
  python -m pip install transformers peft trl sentence-transformers scikit-learn umap-learn pandas numpy tqdm datasets
else
  _hpc_activate_and_fix_path
  _py="$(command -v python 2>/dev/null || true)"
  if [ "${_py}" = "/opt/miniconda3/bin/python" ] || [ "${_py}" = "/opt/miniconda3/bin/python3" ]; then
    echo "[hpc_env_deakin] ERROR: conda activate did not switch off base Python (${_py})."
    echo "  Your env may be broken or in a read-only prefix. Try on the login node:"
    echo "    conda env remove -n ${CONDA_ENV} -y"
    echo "  Then re-submit with CREATE_ENV=1 (env is created under ~/.conda/envs when prepended)."
    exit 1
  fi
  _hpc_ensure_bitsandbytes || exit 1
fi

echo "[hpc_env_deakin] Using $(command -v python) — $(python --version 2>&1) (env=${CONDA_ENV}, CONDA_PREFIX=${CONDA_PREFIX:-unset})"
_hpc_prepend_ld_for_torch
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if ! python -c "import torch; print('[hpc_env_deakin] torch OK:', torch.__version__, 'cuda:', torch.cuda.is_available())"; then
    echo "[hpc_env_deakin] ERROR: PyTorch import failed. On the login node run:"
    echo "  source \"${_CONDA_SH}\" && conda activate ${CONDA_ENV} && python -m pip install torch --index-url https://download.pytorch.org/whl/cu118"
    exit 1
  fi
  if [ "${HPC_REQUIRE_CUDA:-1}" = "1" ]; then
    if ! python - <<'PY'
import os
import sys

import torch

def cuda_ok() -> bool:
    return bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0

# Inside a 1-GPU Slurm step, logical device 0 is correct (not SLURM_JOB_GPUS global index).
for cvd in ("0", os.environ.get("CUDA_VISIBLE_DEVICES", "")):
    if not str(cvd).strip():
        continue
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd).split(",")[0].strip() or "0"
    if cuda_ok():
        name = torch.cuda.get_device_name(0)
        print(f"[hpc_env_deakin] cuda=True device_count={torch.cuda.device_count()} GPU={name}")
        sys.exit(0)

print("[hpc_env_deakin] cuda=False after CUDA_VISIBLE_DEVICES retries", file=sys.stderr)
print(
    f"  SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID','')} "
    f"node={os.environ.get('SLURMD_NODENAME', os.environ.get('SLURM_NODELIST',''))} "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} "
    f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS','')}",
    file=sys.stderr,
)
sys.exit(1)
PY
    then
      echo "[hpc_env_deakin] ERROR: Slurm GPU allocation but torch.cuda.is_available() is False."
      echo "  (Job is already on an allocated node — requeue will not auto-switch host; fix env or pick another node in submit pool.)"
      echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-}  SLURM_NODELIST=${SLURM_NODELIST:-}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
      echo "  HPC_MODULE_CUDA=${HPC_MODULE_CUDA:-<auto>}  HPC_MODULE_PURGE=${HPC_MODULE_PURGE:-0}"
      if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[hpc_env_deakin] nvidia-smi:"
        nvidia-smi -L 2>&1 || nvidia-smi 2>&1 || true
      else
        echo "[hpc_env_deakin] nvidia-smi not found on PATH."
      fi
      if command -v module >/dev/null 2>&1; then
        echo "[hpc_env_deakin] module list:"
        module list 2>&1 || true
        echo "[hpc_env_deakin] module avail cuda (first lines):"
        module avail cuda 2>&1 | head -20 || true
      fi
      echo "[hpc_env_deakin] Fix: on a DGX node run  module avail cuda  then resubmit with e.g."
      echo "  HPC_MODULE_CUDA=CUDA/12.6.0  (use exact name from: module avail cuda on a DGX node)"
      exit 1
    fi
  fi
else
  if ! python -c "import torch; print('[hpc_env_deakin] torch OK:', torch.__version__, '(CUDA checked in Slurm GPU job only)')"; then
    echo "[hpc_env_deakin] ERROR: PyTorch import failed. On the login node run:"
    echo "  source \"${_CONDA_SH}\" && conda activate ${CONDA_ENV} && python -m pip install torch --index-url https://download.pytorch.org/whl/cu118"
    exit 1
  fi
fi

# Remap OUTPUT_DIR / SAVE_ROOT / input CSVs that were already in the environment
# (typical sbatch --export) onto scratch. Jobs that set paths *after* this file
# must call hpc_scratch_auto_bind again.
if [ -n "${SLURM_JOB_ID:-}" ] && command -v hpc_scratch_auto_bind >/dev/null 2>&1; then
  hpc_scratch_auto_bind
fi
