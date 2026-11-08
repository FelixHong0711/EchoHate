#!/usr/bin/env bash
# =============================================================================
# CUDA defaults for all echohate Slurm GPU jobs (sourced from hpc_env_deakin.sh).
# Submit scripts: export via hpc_defaults_from_local.sh and/or echohate_hpc_cuda_export_suffix.
# =============================================================================
#   HPC_REQUIRE_CUDA=1     exit if torch.cuda.is_available() is False (in hpc_env_deakin.sh)
#   HPC_MODULE_PURGE=0     do not `module purge` on compute nodes (breaks DGX CUDA stack)
#   CUDA_VISIBLE_DEVICES   default 0 for single-GPU Slurm 1-GPU jobs (see hpc_env_deakin.sh)
#   HPLOGIN_DGX_PROBE_CUDA_NODES=1  (submit): ssh nvidia-smi on pool nodes; drop dead hosts from nodelist
# =============================================================================

export HPC_REQUIRE_CUDA="${HPC_REQUIRE_CUDA:-1}"
export HPC_MODULE_PURGE="${HPC_MODULE_PURGE:-0}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Always device 0 inside a 1-GPU allocation. Stale sbatch --export=ALL values (e.g. =3)
  # and SLURM_JOB_GPUS (physical index) both break torch.cuda.is_available() on DGX.
  export CUDA_VISIBLE_DEVICES=0
fi
