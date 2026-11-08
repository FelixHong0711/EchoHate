#!/usr/bin/env bash
# Shared defaults for *_from_local.sh and bootstrap_hpc_conda_from_local.sh
# If your shell exports HPC_HOST=hpclogin.deakingpuhpc.deakin.edu.au (old login),
# we remap to the new research HPC unless USE_LEGACY_HPC_LOGIN=1.
#
# Legacy hpclogin + DGX GPUs: use submit_*_hpclogin_dgx_from_local.sh wrappers or export
#   ECHOHATE_HPC_PROFILE=hpclogin_dgx
# before sourcing this file (sets USE_LEGACY_HPC_LOGIN, HPC_HOST, SBATCH_PARTITION, SBATCH_NODES).

HPC_USER="${HPC_USER:-hohuu}"

_LEGACY_LOGIN="hpclogin.deakingpuhpc.deakin.edu.au"
_NEW_LOGIN="hpc2.it.deakin.edu.au"

# Default cluster: hpclogin + DGX unless you target hpc2 (see below).
# Override: export ECHOHATE_HPC_PROFILE=hpc2  OR  export HPC_HOST=hpc2.it.deakin.edu.au
# Note: unset ECHOHATE_HPC_PROFILE alone still defaults to hpclogin_dgx — use ECHOHATE_HPC_PROFILE=hpc2 for hpc2.
export ECHOHATE_HPC_PROFILE="${ECHOHATE_HPC_PROFILE:-hpclogin_dgx}"

# hpclogin DGX pool: **never default to hpc-dgx-b-1** (flaky CUDA). Use b-2,b-3 unless overridden.
_HPLOGIN_DGX_DEFAULT_NODES="hpc-dgx-b-2,hpc-dgx-b-3"

# Strip hpc-dgx-b-1 from a comma list unless HPC_ALLOW_DGX_B1=1 or KEEP_AMBIENT_NODES=1.
echohate_hpclogin_dgx_sanitize_nodes() {
  local csv="${1:-}"
  local n out=()
  if [[ -z "${csv}" ]]; then
    csv="${_HPLOGIN_DGX_DEFAULT_NODES}"
  fi
  if [[ "${HPC_ALLOW_DGX_B1:-0}" != "1" ]] && [[ "${KEEP_AMBIENT_NODES:-0}" != "1" ]]; then
    IFS=',' read -r -a _parts <<< "${csv}"
    for n in "${_parts[@]}"; do
      n="${n// /}"
      [[ -z "${n}" ]] && continue
      [[ "${n}" == "hpc-dgx-b-1" ]] && continue
      out+=("${n}")
    done
    if [[ ${#out[@]} -gt 0 ]]; then
      csv="$(IFS=','; echo "${out[*]}")"
    else
      csv="${_HPLOGIN_DGX_DEFAULT_NODES}"
    fi
  fi
  if [[ "${csv}" == *"hpc-dgx-b-3"* ]]; then
    export SBATCH_ALLOW_DGX_B3=1
  fi
  printf '%s' "${csv}"
}

# Apply sanitized hpclogin DGX node defaults to HPLOGIN_DGX_NODE_CSV and SBATCH_NODES.
echohate_hpclogin_dgx_apply_node_defaults() {
  local _csv
  _csv="$(echohate_hpclogin_dgx_sanitize_nodes "${HPLOGIN_DGX_NODE_CSV:-${SBATCH_NODES:-}}")"
  export HPLOGIN_DGX_NODE_CSV="${_csv}"
  export SBATCH_NODES="$(echohate_hpclogin_dgx_sanitize_nodes "${SBATCH_NODES:-${_csv}}")"
}

# If HPC_HOST already points at hpc2, do not let the hpclogin_dgx default overwrite it.
if [ "${ECHOHATE_HPC_PROFILE}" = "hpclogin_dgx" ] && [ "${HPC_HOST:-}" = "${_NEW_LOGIN}" ]; then
  export ECHOHATE_HPC_PROFILE=hpc2
fi

# Preset: submit jobs to legacy hpclogin with Slurm partition "dgx" (see sinfo: hpc-dgx-b-*).
# Pick the box with --nodelist / SBATCH_NODES (default hpc-dgx-b-2,hpc-dgx-b-3; avoid b-1). Names like "dgx-1" are not Slurm
# partitions on this login; if they were exported from another environment, force "dgx".
if [ "${ECHOHATE_HPC_PROFILE:-}" = "hpclogin_dgx" ]; then
  export USE_LEGACY_HPC_LOGIN=1
  export HPC_HOST="${_LEGACY_LOGIN}"
  case "${SBATCH_PARTITION:-}" in
    dgx-1 | dgx-2 | gpu)
      echo "[job_submission] WARNING: SBATCH_PARTITION=${SBATCH_PARTITION} is invalid on hpclogin Slurm (use partition \"dgx\" + --nodelist hpc-dgx-b-*). Forcing SBATCH_PARTITION=dgx." >&2
      export SBATCH_PARTITION=dgx
      ;;
    *)
      export SBATCH_PARTITION="${SBATCH_PARTITION:-dgx}"
      ;;
  esac
  # Eligible DGX hosts (comma list): Slurm still runs **one** node per job when sbatch passes --nodes=1.
  # Default pool: **b-2,b-3** (b-1 stripped unless HPC_ALLOW_DGX_B1=1). Override: `export HPLOGIN_DGX_NODE_CSV=...`
  export HPLOGIN_DGX_NODE_CSV="${HPLOGIN_DGX_NODE_CSV:-${_HPLOGIN_DGX_DEFAULT_NODES}}"
  echohate_hpclogin_dgx_apply_node_defaults
  echo "[job_submission] ECHOHATE_HPC_PROFILE=hpclogin_dgx → HPC_HOST=${HPC_HOST} SBATCH_PARTITION=${SBATCH_PARTITION:-} SBATCH_NODES=${SBATCH_NODES:-}" >&2
fi

# All GPU Slurm jobs: require CUDA on compute nodes; do not module-purge DGX stacks (see hpc_cuda_slurm_defaults.sh).
export HPC_REQUIRE_CUDA="${HPC_REQUIRE_CUDA:-1}"
export HPC_MODULE_PURGE="${HPC_MODULE_PURGE:-0}"

# Append to sbatch --export=... (works with or without ALL).
echohate_hpc_cuda_export_suffix() {
  local _cuda_mod="${HPC_MODULE_CUDA:-}"
  local _cuda_fb="${HPC_MODULE_CUDA_FALLBACKS:-}"
  if [ -z "${_cuda_mod}" ] && [ "${ECHOHATE_HPC_PROFILE:-}" = "hpclogin_dgx" ]; then
    # Let hpc_env_deakin.sh try CUDA/12.6.0, 12.2.2, … in order (names differ login vs DGX).
    _cuda_mod=""
    _cuda_fb="${_cuda_fb:-CUDA/12.6.0 CUDA/12.2.2 CUDA/12.2.0 CUDA/11.8.0}"
  fi
  if [ -n "${_cuda_mod}" ]; then
    printf ',HPC_REQUIRE_CUDA=%s,HPC_MODULE_PURGE=%s,HPC_MODULE_CUDA=%s' \
      "${HPC_REQUIRE_CUDA:-1}" "${HPC_MODULE_PURGE:-0}" "${_cuda_mod}"
  else
    printf ',HPC_REQUIRE_CUDA=%s,HPC_MODULE_PURGE=%s' \
      "${HPC_REQUIRE_CUDA:-1}" "${HPC_MODULE_PURGE:-0}"
  fi
  if [ -n "${_cuda_fb}" ]; then
    # Spaces break Slurm --export=ALL,K=v parsing; use '+' (hpc_env expands '+' → space).
    _cuda_fb_export="${_cuda_fb// /+}"
    printf ',HPC_MODULE_CUDA_FALLBACKS=%s' "${_cuda_fb_export}"
  fi
}

# sbatch reads --partition from SBATCH_EXTRA; fix the same dgx-1 / dgx-2 footgun there.
if [[ "${SBATCH_EXTRA:-}" == *"--partition=dgx-1"* ]] || [[ "${SBATCH_EXTRA:-}" == *"--partition=dgx-2"* ]]; then
  echo "[job_submission] WARNING: SBATCH_EXTRA contained --partition=dgx-1 or dgx-2; replacing with --partition=dgx." >&2
  SBATCH_EXTRA="${SBATCH_EXTRA//--partition=dgx-1/--partition=dgx}"
  SBATCH_EXTRA="${SBATCH_EXTRA//--partition=dgx-2/--partition=dgx}"
  export SBATCH_EXTRA
fi

# hpc2: partition "dgx" does not exist (hpclogin only). Stale shell exports often leak here.
if [ "${ECHOHATE_HPC_PROFILE:-}" != "hpclogin_dgx" ] && [ "${USE_LEGACY_HPC_LOGIN:-0}" != "1" ]; then
  if [[ "${SBATCH_EXTRA:-}" == *"--partition=dgx"* ]]; then
    echo "[job_submission] WARNING: SBATCH_EXTRA has --partition=dgx but target is hpc2 (use gpu). Replacing with --partition=gpu." >&2
    SBATCH_EXTRA="${SBATCH_EXTRA//--partition=dgx/--partition=gpu}"
    export SBATCH_EXTRA
  fi
  case "${SBATCH_PARTITION:-}" in
    dgx | dgx-1 | dgx-2)
      echo "[job_submission] WARNING: SBATCH_PARTITION=${SBATCH_PARTITION} invalid on hpc2; forcing gpu." >&2
      export SBATCH_PARTITION=gpu
      ;;
  esac
  if [[ "${SBATCH_EXTRA:-}" == *"hpc-dgx-b-"* ]] || [[ "${SBATCH_NODES:-}" == hpc-dgx-b-* ]]; then
    echo "[job_submission] ERROR: nodelist uses hpclogin hosts (hpc-dgx-b-*) but target is hpc2 (use g46-*). Unset SBATCH_NODES or use HPC2_DEFAULT_NODE=g46-1gpu-4." >&2
    return 1 2>/dev/null || exit 1
  fi
fi

if [ -n "${HPC_HOST:-}" ] && [ "${HPC_HOST}" = "${_LEGACY_LOGIN}" ] && [ "${USE_LEGACY_HPC_LOGIN:-0}" != "1" ]; then
  echo "[job_submission] HPC_HOST was set to the old login (${_LEGACY_LOGIN}); using ${_NEW_LOGIN} instead." >&2
  echo "[job_submission] To silence this: unset HPC_HOST, or export HPC_HOST=${_NEW_LOGIN}, or USE_LEGACY_HPC_LOGIN=1 for the old host." >&2
  HPC_HOST="${_NEW_LOGIN}"
elif [ -z "${HPC_HOST:-}" ]; then
  if [ "${ECHOHATE_HPC_PROFILE:-}" = "hpclogin_dgx" ] || [ "${USE_LEGACY_HPC_LOGIN:-0}" = "1" ]; then
    export USE_LEGACY_HPC_LOGIN=1
    HPC_HOST="${_LEGACY_LOGIN}"
  else
    HPC_HOST="${_NEW_LOGIN}"
  fi
fi

# Portable UTC timestamp for log lines (POSIX `date`; GNU-only `date -Iseconds` is avoided).
hpc_iso_timestamp() {
  LC_ALL=C date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u
}

# Save/restore errexit when helpers are sourced into an interactive shell (avoid closing the terminal tab).
_hpc_errexit_snapshot() {
  case $- in *e*) echo 1 ;; *) echo 0 ;; esac
}
_hpc_errexit_restore() {
  if [ "${1:-0}" -eq 1 ]; then set -e; else set +e; fi
}

# Non-interactive SSH password (same pattern as hpc2 automation): install sshpass locally, then either
#   export HPC_PASSWORD='...'
# or (sshpass default)
#   export SSHPASS='...'
# Uses sshpass -e so the password is not passed on the sshpass argv. Prefer SSH keys when possible.
_hpc_sshpass_secret() {
  if [ -n "${HPC_PASSWORD:-}" ]; then
    printf '%s' "${HPC_PASSWORD}"
  elif [ -n "${SSHPASS:-}" ]; then
    printf '%s' "${SSHPASS}"
  else
    printf '%s' ""
  fi
}

# Non-rsync SSH (mkdir, sbatch, remote $HOME probes via _hpc_ssh): use timeouts so unreachable hosts
# fail instead of hanging. Override before sourcing: export HPC_SSH_CONN_OPTS='-o ConnectTimeout=90 ...'
HPC_SSH_CONN_OPTS="${HPC_SSH_CONN_OPTS:--o ConnectTimeout=45 -o ConnectionAttempts=3 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o TCPKeepAlive=yes}"

_hpc_ssh() {
  local _sec
  local -a _argv _opts
  local mux_dir
  _sec="$(_hpc_sshpass_secret)"
  _argv=(ssh -o GSSAPIAuthentication=no -o IPQoS=none)
  if [ -n "${HPC_SSH_CONN_OPTS:-}" ]; then
    # shellcheck disable=SC2206
    _opts=(${HPC_SSH_CONN_OPTS})
    _argv+=("${_opts[@]}")
  else
    _argv+=(
      -o ConnectTimeout=45
      -o ConnectionAttempts=3
      -o ServerAliveInterval=15
      -o ServerAliveCountMax=8
      -o TCPKeepAlive=yes
    )
  fi
  if [ "${HPC_SSH_RSYNC_IPV4:-0}" = "1" ]; then
    _argv+=(-4)
  fi
  if [ "${HPC_SSH_SUBMIT_MUX:-0}" = "1" ] || [ "${HPC_SSH_SUBMIT_MUX:-}" = "true" ]; then
    mux_dir="$(_hpc_ssh_mux_dir_for_rsync)"
    mkdir -p "${mux_dir}"
    chmod 700 "${mux_dir}" 2>/dev/null || true
    _argv+=(-o ControlMaster=auto "-o ControlPersist=${HPC_SSH_MUX_PERSIST}" "-o ControlPath=${mux_dir}/%C")
  fi
  if [ -n "${_sec}" ] && command -v sshpass >/dev/null 2>&1; then
    SSHPASS="${_sec}" sshpass -e "${_argv[@]}" "$@"
  elif [ -n "${_sec}" ]; then
    echo "[job_submission] WARNING: HPC_PASSWORD/SSHPASS is set but sshpass is not in PATH; using plain ssh (expect a password prompt)." >&2
    "${_argv[@]}" "$@"
  else
    "${_argv[@]}" "$@"
  fi
}

# Used only by hpc_mkdir_remote_from_local. sshpass + passphrase-protected keys = indefinite wait (sshpass
# looks for "assword:" while ssh asks for a key passphrase). Skip sshpass unless HPC_SSHPASS_FOR_MKDIR=1
# (password-only SSH). Extra opts reduce macOS/home-network stalls (GSSAPI, IPQoS). Do not use -tt with
# merged stderr→file — passphrase prompts must reach the real terminal (see hpc_mkdir_remote_from_local).
_hpc_ssh_for_mkdir() {
  local _sec
  local -a _argv _opts
  local mux_dir
  _sec="$(_hpc_sshpass_secret)"
  _argv=(ssh -o GSSAPIAuthentication=no -o IPQoS=none)
  if [ -n "${HPC_SSH_CONN_OPTS:-}" ]; then
    # shellcheck disable=SC2206
    _opts=(${HPC_SSH_CONN_OPTS})
    _argv+=("${_opts[@]}")
  else
    _argv+=(
      -o ConnectTimeout=45
      -o ConnectionAttempts=3
      -o ServerAliveInterval=15
      -o ServerAliveCountMax=8
      -o TCPKeepAlive=yes
    )
  fi
  if [ "${HPC_SSH_RSYNC_IPV4:-0}" = "1" ]; then
    _argv+=(-4)
  fi
  if [ "${HPC_SSH_SUBMIT_MUX:-0}" = "1" ] || [ "${HPC_SSH_SUBMIT_MUX:-}" = "true" ]; then
    mux_dir="$(_hpc_ssh_mux_dir_for_rsync)"
    mkdir -p "${mux_dir}"
    chmod 700 "${mux_dir}" 2>/dev/null || true
    _argv+=(-o ControlMaster=auto "-o ControlPersist=${HPC_SSH_MUX_PERSIST}" "-o ControlPath=${mux_dir}/%C")
  fi
  if [ -n "${_sec}" ] && command -v sshpass >/dev/null 2>&1 && { [ "${HPC_SSHPASS_FOR_MKDIR:-0}" = "1" ] || [ "${HPC_SSHPASS_FOR_MKDIR:-}" = "true" ]; }; then
    SSHPASS="${_sec}" sshpass -e "${_argv[@]}" "$@"
  else
    if [ -n "${_sec}" ] && command -v sshpass >/dev/null 2>&1; then
      echo "[job_submission] mkdir: ignoring sshpass (set HPC_SSHPASS_FOR_MKDIR=1 only for password-only SSH)." >&2
    fi
    "${_argv[@]}" "$@"
  fi
}

# Grid submitters call _hpc_ssh once per sbatch. Fresh password auth each time often hits rate limits → mid-loop
# "Permission denied (publickey,password)". Default HPC_SSH_SUBMIT_MUX=1 when using sshpass so one ControlMaster
# carries many sbatch invocations (override: export HPC_SSH_SUBMIT_MUX=0).
if { [ -n "${HPC_PASSWORD:-}" ] || [ -n "${SSHPASS:-}" ]; } && command -v sshpass >/dev/null 2>&1; then
  export HPC_SSH_SUBMIT_MUX="${HPC_SSH_SUBMIT_MUX:-1}"
fi

# Post-rsync sha256 check (submit_llm_from_local.sh, submit_abstract_from_local.sh, …).
# Uses _hpc_ssh so HPC_PASSWORD / SSHPASS + sshpass work (same as sbatch). Skip: HPC_SKIP_VERIFY_SYNC=1.
echohate_local_sha256() {
  local f="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  else
    python3 - "$f" <<'PY'
import hashlib, sys
p = sys.argv[1]
h = hashlib.sha256()
with open(p, "rb") as fh:
    for ch in iter(lambda: fh.read(1024 * 1024), b""):
        h.update(ch)
print(h.hexdigest())
PY
  fi
}

echohate_remote_sha256() {
  local remote_file="$1"
  _hpc_ssh "${HPC_USER}@${HPC_HOST}" "if command -v sha256sum >/dev/null 2>&1; then sha256sum '${remote_file}' | awk '{print \$1}'; elif command -v shasum >/dev/null 2>&1; then shasum -a 256 '${remote_file}' | awk '{print \$1}'; else python3 - <<'PY'
import hashlib
p='${remote_file}'
h=hashlib.sha256()
with open(p,'rb') as fh:
    for ch in iter(lambda: fh.read(1024*1024), b''):
        h.update(ch)
print(h.hexdigest())
PY
fi"
}

# echohate_verify_synced_files_or_exit LOCAL_ECHOHATE_ROOT HPC_ECHOHATE_ROOT rel/path [...]
echohate_verify_synced_files_or_exit() {
  local local_root="${1:?local echohate root}"
  local remote_root="${2:?remote echohate root (e.g. \$HPC_RA/experiment/echohate)}"
  shift 2
  if [ "${HPC_SKIP_VERIFY_SYNC:-0}" = "1" ]; then
    echo "[job_submission] HPC_SKIP_VERIFY_SYNC=1 — skipping sha256 verify." >&2
    return 0
  fi
  if [ "$#" -eq 0 ]; then
    echo "[job_submission] echohate_verify_synced_files_or_exit: no relative paths given" >&2
    return 1
  fi
  if [ -n "$(_hpc_sshpass_secret)" ] && command -v sshpass >/dev/null 2>&1; then
    echo "[job_submission] Verifying remote matches local (sha256, $# files, sshpass)..." >&2
  else
    echo "[job_submission] Verifying remote matches local (sha256, $# files)..." >&2
  fi
  local rel lf rf lsum rsum
  for rel in "$@"; do
    lf="${local_root}/${rel}"
    rf="${remote_root}/${rel}"
    if [ ! -f "${lf}" ]; then
      echo "❌ Local file missing for verification: ${lf}" >&2
      return 1
    fi
    lsum="$(echohate_local_sha256 "${lf}")"
    rsum="$(echohate_remote_sha256 "${rf}" 2>/dev/null || true)"
    if [ -z "${rsum}" ]; then
      echo "❌ Remote file missing/unreadable: ${rf}" >&2
      return 1
    fi
    if [ "${lsum}" != "${rsum}" ]; then
      echo "❌ Sync verification failed for ${rel}" >&2
      echo "   local:  ${lsum}" >&2
      echo "   remote: ${rsum}" >&2
      echo "   Refusing to submit to avoid running stale code." >&2
      return 1
    fi
  done
  echo "✅ Verification passed: remote matches local for critical files." >&2
  return 0
}

# Rsync over SSH: keepalives + partial files + retries reduce "protocol data stream (code 12)" when VPN/login drops mid-transfer.
#
# Deakin SSH: if HPC_SSH_RSYNC_IPV4 is unset, default 1 for *.deakin.edu.au (ssh -4). Override: export HPC_SSH_RSYNC_IPV4=0
# Deakin: if HPC_SSH_MUX is unset, default 0 (fresh ssh per rsync; mux + flaky VPN often repeats code 12). Override: export HPC_SSH_MUX=1
# Deakin: if HPC_RSYNC_CODE_FLAGS is unset, default -av (no -z) for lighter code-only syncs. Override: export HPC_RSYNC_CODE_FLAGS='-avz'
# Still failing? export HPC_SKIP_CODE_RSYNC=1 in submit_*_from_local.sh callers after manual sync; HPC_SSH_RSYNC_EXTRA_OPTS='-o IPQoS=none'; wired link.
#
# Override entirely by exporting HPC_SSH_RSYNC_OPTS before sourcing this file.
# Default does not set BatchMode=yes so rsync's ssh can use interactive password auth from a TTY.
# For unattended/CI (keys only), export e.g. HPC_SSH_RSYNC_OPTS='-o BatchMode=yes -o ServerAliveInterval=10 ...' before sourcing.
#
# Many sequential ssh calls (e.g. grid submit → one sbatch per ssh): export HPC_SSH_SUBMIT_MUX=1 so _hpc_ssh
# reuses one ControlMaster connection (one password / key handshake). Independent of HPC_SSH_MUX (rsync).
# When HPC_PASSWORD/SSHPASS + sshpass are used, HPC_SSH_SUBMIT_MUX defaults to 1 (see just after _hpc_ssh).
HPC_SSH_RSYNC_OPTS="${HPC_SSH_RSYNC_OPTS:--o ServerAliveInterval=10 -o ServerAliveCountMax=24 -o TCPKeepAlive=yes -o ExitOnForwardFailure=no -o ConnectTimeout=45 -o ConnectionAttempts=3}"
HPC_SSH_RSYNC_EXTRA_OPTS="${HPC_SSH_RSYNC_EXTRA_OPTS:-}"
HPC_RSYNC_RETRIES="${HPC_RSYNC_RETRIES:-24}"
HPC_RSYNC_RETRY_SLEEP_SEC="${HPC_RSYNC_RETRY_SLEEP_SEC:-12}"
# If 1, backoff sleep increases with each retry (code 12 / other transient errors).
HPC_RSYNC_RETRY_BACKOFF="${HPC_RSYNC_RETRY_BACKOFF:-1}"
# Cap backoff sleep (seconds); raise if login node / VPN needs longer recovery pauses.
HPC_RSYNC_RETRY_MAX_SLEEP_SEC="${HPC_RSYNC_RETRY_MAX_SLEEP_SEC:-300}"
# Extra rsync I/O timeouts (seconds). Raise HPC_RSYNC_TIMEOUT if huge files sit on slow disks.
HPC_RSYNC_TIMEOUT="${HPC_RSYNC_TIMEOUT:-900}"
HPC_RSYNC_CONTIMEOUT="${HPC_RSYNC_CONTIMEOUT:-180}"
# Hidden dir on receiver for incomplete files (resume-friendly).
HPC_RSYNC_PARTIAL_DIR="${HPC_RSYNC_PARTIAL_DIR:-.rsync_partial_hpc}"
# Optional KiB/s cap (integer): export HPC_RSYNC_BWLIMIT_KB=3000
HPC_RSYNC_BWLIMIT_KB="${HPC_RSYNC_BWLIMIT_KB:-}"
# SSH ControlMaster (unset = auto: off for *.deakin.edu.au, on elsewhere). Explicit 0/1 always wins.
if [ -z "${HPC_SSH_MUX+x}" ]; then
  HPC_SSH_MUX=1
  case "${HPC_HOST}" in *deakin.edu.au*) HPC_SSH_MUX=0 ;; esac
fi
HPC_SSH_MUX_DIR="${HPC_SSH_MUX_DIR:-}"
# How long the ControlMaster socket stays alive after the last session (seconds). One successful auth
# (e.g. sshpass password) can then serve repeated ssh/rsync/sbatch until this expires. Shorten before source: HPC_SSH_MUX_PERSIST=600
HPC_SSH_MUX_PERSIST="${HPC_SSH_MUX_PERSIST:-7200}"
HPC_SSH_MUX_RESET_ON_RSYNC_12="${HPC_SSH_MUX_RESET_ON_RSYNC_12:-1}"
# If 1, pass ssh -4 (IPv4 only). Unset = auto: 1 for *.deakin.edu.au else 0. Explicit 0/1 always wins.
if [ -z "${HPC_SSH_RSYNC_IPV4+x}" ]; then
  HPC_SSH_RSYNC_IPV4=0
  case "${HPC_HOST}" in *deakin.edu.au*) HPC_SSH_RSYNC_IPV4=1 ;; esac
fi

# rsync archive/compress flags for code-only trees (used by some *_from_local.sh). Unset = auto: -avz, or -av on *.deakin.edu.au.
if [ -z "${HPC_RSYNC_CODE_FLAGS+x}" ]; then
  HPC_RSYNC_CODE_FLAGS="-avz"
  case "${HPC_HOST}" in *deakin.edu.au*) HPC_RSYNC_CODE_FLAGS="-av" ;; esac
fi

_hpc_ssh_rsync_ipv4_prefix() {
  if [ "${HPC_SSH_RSYNC_IPV4:-0}" = "1" ]; then
    printf '%s' "-4 "
  else
    printf '%s' ""
  fi
}

_hpc_ssh_mux_dir_for_rsync() {
  if [ -n "${HPC_SSH_MUX_DIR:-}" ]; then
    printf '%s' "${HPC_SSH_MUX_DIR}"
    return 0
  fi
  # macOS OpenSSH limits ControlPath to ~104 bytes. mux_<long-hostname>_<user>/%C exceeds that.
  printf '%s' "${HOME}/.ssh/cm"
}

_hpc_ssh_mux_opts_for_rsync() {
  if [ "${HPC_SSH_MUX:-0}" != "1" ]; then
    printf '%s' ""
    return 0
  fi
  local mux_dir
  mux_dir="$(_hpc_ssh_mux_dir_for_rsync)"
  mkdir -p "${mux_dir}"
  chmod 700 "${mux_dir}" 2>/dev/null || true
  printf '%s' "-o ControlMaster=auto -o ControlPersist=${HPC_SSH_MUX_PERSIST} -o ControlPath=${mux_dir}/%%C "
}

# Reused by _hpc_ssh when HPC_SSH_SUBMIT_MUX=1 (grid sbatch loops); same socket dir as rsync mux for one host/user.
_hpc_ssh_cmd_mux_opts() {
  if [ "${HPC_SSH_SUBMIT_MUX:-0}" != "1" ] && [ "${HPC_SSH_SUBMIT_MUX:-}" != "true" ]; then
    printf '%s' ""
    return 0
  fi
  local mux_dir
  mux_dir="$(_hpc_ssh_mux_dir_for_rsync)"
  mkdir -p "${mux_dir}"
  chmod 700 "${mux_dir}" 2>/dev/null || true
  printf '%s' "-o ControlMaster=auto -o ControlPersist=${HPC_SSH_MUX_PERSIST} -o ControlPath=${mux_dir}/%%C "
}

_hpc_ssh_mux_close_sockets() {
  [ "${HPC_SSH_MUX:-0}" != "1" ] && return 0
  local mux_dir ctl
  mux_dir="$(_hpc_ssh_mux_dir_for_rsync)"
  [ -d "${mux_dir}" ] || return 0
  shopt -s nullglob
  for ctl in "${mux_dir}"/*; do
    [ -S "${ctl}" ] || continue
    # Plain ssh: do not use _hpc_ssh (avoids nested ControlMaster / sshpass on mux -O exit).
    command ssh $(_hpc_ssh_rsync_ipv4_prefix) -o BatchMode=yes -S "${ctl}" -O exit "${HPC_USER}@${HPC_HOST}" 2>/dev/null || true
  done
  shopt -u nullglob
}

# Usage: hpc_rsync_with_retry -avz --progress --exclude=... src/ user@host:dst/
# Does not pass --delete: destination files missing from the source are left as-is unless callers add --delete.
# HPC_RSYNC_USE_PLAIN_SSH=1: skip sshpass for rsync's ssh (passphrase keys + sshpass deadlock; MSI submitters set this).
hpc_rsync_with_retry() {
  local attempt=1
  local max="${HPC_RSYNC_RETRIES}"
  local delay="${HPC_RSYNC_RETRY_SLEEP_SEC}"
  local rc=0
  local mux_opts ipv4pfx
  local _errexit_was
  _errexit_was="$(_hpc_errexit_snapshot)"
  mux_opts="$(_hpc_ssh_mux_opts_for_rsync)"
  ipv4pfx="$(_hpc_ssh_rsync_ipv4_prefix)"
  local -a rs_base
  rs_base=(
    rsync
    --partial
    --timeout="${HPC_RSYNC_TIMEOUT}"
    --contimeout="${HPC_RSYNC_CONTIMEOUT}"
  )
  if [ -n "${HPC_RSYNC_PARTIAL_DIR:-}" ]; then
    rs_base+=(--partial-dir="${HPC_RSYNC_PARTIAL_DIR}")
  fi
  if [ -n "${HPC_RSYNC_BWLIMIT_KB:-}" ]; then
    rs_base+=(--bwlimit="${HPC_RSYNC_BWLIMIT_KB}")
  fi
  local _sec _rsh _use_sshpass
  _sec="$(_hpc_sshpass_secret)"
  _use_sshpass=0
  if [ "${HPC_RSYNC_USE_PLAIN_SSH:-0}" = "1" ] || [ "${HPC_RSYNC_USE_PLAIN_SSH:-}" = "true" ]; then
    _sec=""
  fi
  if [ -n "${_sec}" ] && command -v sshpass >/dev/null 2>&1; then
    _use_sshpass=1
  elif [ -n "${_sec}" ]; then
    echo "[job_submission] WARNING: HPC_PASSWORD/SSHPASS set but sshpass missing; rsync may prompt for password." >&2
  fi
  _rsh="ssh -T ${ipv4pfx}${mux_opts}${HPC_SSH_RSYNC_OPTS} ${HPC_SSH_RSYNC_EXTRA_OPTS}"
  while [ "${attempt}" -le "${max}" ]; do
    set +e
    if [ "${_use_sshpass}" -eq 1 ]; then
      SSHPASS="${_sec}" sshpass -e "${rs_base[@]}" -e "${_rsh}" "$@"
    else
      "${rs_base[@]}" -e "${_rsh}" "$@"
    fi
    rc=$?
    _hpc_errexit_restore "${_errexit_was}"
    if [ "${rc}" -eq 0 ]; then
      return 0
    fi
    if [ "${rc}" -eq 23 ]; then
      echo "[job_submission] rsync exit 23: missing source path(s), vanished files, or partial transfer — check remote paths; retries will not create missing files." >&2
      return 23
    fi
    if [ "${rc}" -eq 12 ]; then
      echo "[job_submission] rsync exit 12: protocol/stream error (often SSH/VPN dropped mid-transfer)." >&2
      if [ "${HPC_SSH_MUX_RESET_ON_RSYNC_12:-1}" = "1" ] && [ "${HPC_SSH_MUX:-0}" = "1" ]; then
        echo "[job_submission] recovery: closing stale SSH mux sockets under $(_hpc_ssh_mux_dir_for_rsync), then retrying..." >&2
        _hpc_ssh_mux_close_sockets
      fi
    else
      echo "[job_submission] rsync failed (exit ${rc})." >&2
    fi
    if [ "${attempt}" -ge "${max}" ]; then
      echo "[job_submission] rsync: exhausted ${max} attempt(s). Try HPC_SKIP_CODE_RSYNC=1 (after manual sync), HPC_RSYNC_BWLIMIT_KB=2500, HPC_SSH_RSYNC_EXTRA_OPTS='-o IPQoS=none', HPC_RSYNC_CODE_FLAGS='-av', or sync from login node." >&2
      return "${rc}"
    fi
    local sleep_sec="${delay}"
    local cap="${HPC_RSYNC_RETRY_MAX_SLEEP_SEC:-300}"
    if [ "${HPC_RSYNC_RETRY_BACKOFF:-0}" = "1" ]; then
      sleep_sec=$((delay * attempt))
      if [ "${sleep_sec}" -gt "${cap}" ]; then
        sleep_sec="${cap}"
      fi
    fi
    echo "[job_submission] retry ${attempt}/${max} in ${sleep_sec}s ..." >&2
    attempt=$((attempt + 1))
    sleep "${sleep_sec}"
  done
  _hpc_errexit_restore "${_errexit_was}"
  return "${rc}"
}

# Non-interactive `ssh user@host "cmd"` often gets a minimal PATH — Slurm may not be found (sbatch/squeue).
# Prepend common locations; override with HPC_SLURM_PATH or set HPC_SBATCH to the full path to sbatch.
HPC_SLURM_PATH="${HPC_SLURM_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/slurm/bin:/opt/slurm/current/bin}"
HPC_SBATCH="${HPC_SBATCH:-sbatch}"

# shellcheck disable=SC2016
hpc_remote_slurm_env() {
  printf 'export PATH="%s:$PATH"' "${HPC_SLURM_PATH}"
}

# Set HPC_RA from the login node's real home/RA (one SSH).
# Prefer passwd(5) home via getent: some sessions (IDE remote, profile quirks) set $HOME to
# /home/remote/... while Slurm/rsync and real data live under the passwd home (e.g. /home/hohuu).
_hpc_set_hpc_ra_from_remote_home() {
  local home _rc _errexit_was
  # Do not leak `set -e` into the caller's interactive shell (was: set -e mid-function with no restore).
  _errexit_was=0
  case $- in *e*) _errexit_was=1 ;; esac
  set +e
  home=$(
    _hpc_ssh "${HPC_USER}@${HPC_HOST}" "getent passwd '${HPC_USER}' 2>/dev/null | cut -d: -f6"
  )
  _rc=$?
  if [ "${_rc}" -ne 0 ] || [ -z "${home}" ]; then
    home=$(_hpc_ssh "${HPC_USER}@${HPC_HOST}" 'echo $HOME') || {
      echo "[job_submission] ERROR: ssh failed; set HPC_RA manually to \\\$HOME/RA on the login node." >&2
      if [ "${_errexit_was}" -eq 1 ]; then set -e; else set +e; fi
      return 1
    }
  fi
  HPC_RA="${home}/RA"
  echo "[job_submission] HPC_RA=${HPC_RA}  (resolved passwd/getent home when available)" >&2
  if echo "${HPC_RA}" | grep -q '/home/remote' && [ "${ECHOHATE_HPC_PROFILE:-}" != "hpclogin_dgx" ]; then
    echo "[job_submission] WARNING: HPC_RA still under /home/remote — override with: export HPC_RA=/home/${HPC_USER}/RA  (or your real cluster home)." >&2
  fi
  if [ "${_errexit_was}" -eq 1 ]; then set -e; else set +e; fi
  return 0
}

# Call right after sourcing this file (before HPC_DIR=...).
#
# Persistent project root (Julian / cluster policy):
#   Default HPC_RA=/scratch/$USER/RA  — NOT $HOME (home quota + NFS thrash).
# Job-local high-I/O (DGX NVMe):
#   /raid/$SLURM_JOB_ID via job_submission/hpc_dgx_scratch.sh (stage-in / stage-out).
#
# Escape hatches:
#   HPC_RA_FORCE_HOME=1  — resolve remote $HOME/RA (emergencies only)
#   HPC_RA_NO_AUTO_FIX=1 — keep an explicit /home/... HPC_RA you exported
#   HPC_RA_RESOLVE=1     — legacy flag; now also prefers /scratch (use FORCE_HOME for home)
hpc_apply_ra_default_or_resolve() {
  local _scratch_default="/scratch/${HPC_USER:-hohuu}/RA"
  if [ "${HPC_RA_FORCE_HOME:-0}" = "1" ]; then
    echo "[job_submission] HPC_RA_FORCE_HOME=1: using remote \\\$HOME/RA (not recommended for ML I/O) ..." >&2
    _hpc_set_hpc_ra_from_remote_home || return 1
  else
    if [ -z "${HPC_RA:-}" ] || [ "${HPC_RA}" = "/home/hohuu/RA" ] || [ "${HPC_RA}" = "/home/${HPC_USER:-hohuu}/RA" ]; then
      HPC_RA="${_scratch_default}"
    fi
    if [ "${HPC_RA_NO_AUTO_FIX:-0}" != "1" ] && echo "${HPC_RA}" | grep -qE '^/home/'; then
      echo "[job_submission] WARNING: HPC_RA=${HPC_RA} is under home — rewriting to ${_scratch_default}." >&2
      echo "[job_submission]   (Julian: use /scratch for durable temps + /raid/\$SLURM_JOB_ID for DGX I/O; set HPC_RA_FORCE_HOME=1 to keep home.)" >&2
      HPC_RA="${_scratch_default}"
    fi
    if [ "${HPC_RA_RESOLVE:-0}" = "1" ]; then
      echo "[job_submission] HPC_RA_RESOLVE=1 ignored for home; durable root is cluster scratch: ${HPC_RA}" >&2
      echo "[job_submission]   (set HPC_RA_FORCE_HOME=1 if you truly need \\\$HOME/RA)" >&2
    fi
  fi
  echo "[job_submission] HPC_RA=${HPC_RA}" >&2
}

# Create directories on the login node before rsync. Paths must exist under your real $HOME on the HPC.
# Usage: hpc_mkdir_remote_from_local "${HPC_DIR}" "${HPC_DIR}/data/${DATASET}"
#
# Important: do not run ssh inside command substitution (no TTY / wrong stdin for passphrases). Capture
# stdout to a temp file only; send **stderr to /dev/tty** when available so passphrase / hostkey prompts
# are visible (2>&1 into the file hides prompts and looks like an infinite hang).
hpc_mkdir_remote_from_local() {
  if [ "$#" -eq 0 ]; then
    echo "hpc_mkdir_remote_from_local: no paths" >&2
    return 1
  fi
  local q _out _rc _tfile _errexit_was
  _errexit_was="$(_hpc_errexit_snapshot)"
  q=$(printf '%q ' "$@")
  _tfile="$(mktemp "${TMPDIR:-/tmp}/hpc_mkdir_ssh.XXXXXX" 2>/dev/null || echo "${TMPDIR:-/tmp}/hpc_mkdir_ssh_$$")"
  echo "[job_submission] mkdir: connecting ${HPC_USER}@${HPC_HOST} (passphrase prompts appear below if needed)..." >&2
  set +e
  # Prefer /dev/tty when it is writable (interactive). Cursor/agent shells often have
  # a char-device /dev/tty that still rejects writes ("Device not configured").
  if [ -c /dev/tty ] 2>/dev/null && { : >/dev/tty; } 2>/dev/null; then
    _hpc_ssh_for_mkdir "${HPC_USER}@${HPC_HOST}" "mkdir -p $q" >"${_tfile}" 2>/dev/tty
  else
    _hpc_ssh_for_mkdir "${HPC_USER}@${HPC_HOST}" "mkdir -p $q" >"${_tfile}" 2>&1
  fi
  _rc=$?
  _hpc_errexit_restore "${_errexit_was}"
  _out=""
  if [ -s "${_tfile}" ]; then
    _out="$(cat "${_tfile}")"
  fi
  rm -f "${_tfile}" 2>/dev/null || true
  if [ "${_rc}" -ne 0 ]; then
    echo "" >&2
    echo "ERROR: could not mkdir on HPC (ssh exit ${_rc})." >&2
    if [ -n "${_out}" ]; then
      echo "  Remote/ssh said: ${_out}" >&2
    fi
    echo "  Current HPC_RA=${HPC_RA:-unset}  (paths were under \\\$HPC_RA on the cluster)" >&2
    echo "  If you see 'Permission denied' on /home/remote, your HPC_RA is wrong (IDE remote paths are not the cluster home)." >&2
    echo "  Fix (pick one):" >&2
    echo "    unset HPC_RA && HPC_RA_RESOLVE=1 bash <submit_script>.sh ..." >&2
    echo "    export HPC_RA=\$(ssh ${HPC_USER}@${HPC_HOST} 'echo \$HOME')/RA" >&2
    echo "    Non-interactive (hpc2-style): export HPC_PASSWORD='...' or SSHPASS='...' plus sshpass on PATH; submit scripts use this automatically." >&2
    echo "    mkdir still hangs? Try: unset SSHPASS HPC_PASSWORD (sshpass + key passphrase deadlocks), or HPC_SSH_RSYNC_IPV4=0, or ssh-add." >&2
    return 1
  fi
}

# Most *_from_local.sh jobs: sbatch --export=ALL,... Non-interactive ssh often lacks sbatch on PATH.
# Optional 4th arg: extra sbatch flags before --export (e.g. --job-name=foo or --dependency=afterok:1:2:3).
# Optional 5th arg: shell snippet run on the remote before "cd ..." (e.g. "unset CUDA_VISIBLE_DEVICES"
#   so sbatch --export=ALL does not inherit a stale CUDA_VISIBLE_DEVICES from the ssh session).
# Prints remote sbatch output to stderr; sets HPC_LAST_SSH_JOB_ID; returns 0/1.
# Usage: hpc_ssh_sbatch_export_all "${HPC_DIR}" "${SLURM_SCRIPT}" "${EXPORT_VARS}" [SBATCH_EXTRA] [REMOTE_PREFIX]
hpc_ssh_sbatch_export_all() {
  local HPC_DIR="$1"
  local SLURM_SCRIPT="$2"
  local EXPORT_VARS="$3"
  local SBATCH_EXTRA="${4:-}"
  local REMOTE_PREFIX="${5:-}"
  EXPORT_VARS="${EXPORT_VARS}$(echohate_hpc_cuda_export_suffix)"
  local _SLURM_ENV JOB_LINE _ssh_rc JOB_ID
  local _rp=""
  local _blc_prefix=""

  HPC_LAST_SSH_JOB_ID=""
  _SLURM_ENV="$(hpc_remote_slurm_env)"
  if [ -n "${REMOTE_PREFIX}" ]; then
    _rp="${REMOTE_PREFIX}; "
    _blc_prefix="${REMOTE_PREFIX}; "
  fi
  set +e
  JOB_LINE=$(
    _hpc_ssh "${HPC_USER}@${HPC_HOST}" \
      "${_SLURM_ENV}; ${_rp}cd \"${HPC_DIR}\" && chmod +x \"${SLURM_SCRIPT}\" && \"${HPC_SBATCH}\" --parsable --export=\"${EXPORT_VARS}\" ${SBATCH_EXTRA} \"${SLURM_SCRIPT}\"" 2>&1
  )
  _ssh_rc=$?
  set -e
  echo "${JOB_LINE}" >&2

  JOB_ID=$(echo "${JOB_LINE}" | head -1 | cut -d';' -f1 | tr -d '[:space:]')
  if [ "${_ssh_rc}" -eq 0 ] && [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    HPC_LAST_SSH_JOB_ID="${JOB_ID}"
    return 0
  elif echo "${JOB_LINE}" | grep -qiE 'unrecognised option|unrecognized option|invalid option.*parsable'; then
    echo "Retrying sbatch without --parsable (older Slurm)..." >&2
    set +e
    JOB_LINE=$(
      _hpc_ssh "${HPC_USER}@${HPC_HOST}" \
        "${_SLURM_ENV}; ${_rp}cd \"${HPC_DIR}\" && chmod +x \"${SLURM_SCRIPT}\" && \"${HPC_SBATCH}\" --export=\"${EXPORT_VARS}\" ${SBATCH_EXTRA} \"${SLURM_SCRIPT}\"" 2>&1
    )
    _ssh_rc=$?
    set -e
    echo "${JOB_LINE}" >&2
    JOB_ID=$(echo "${JOB_LINE}" | grep -oE '[0-9]+' | head -1)
  elif echo "${JOB_LINE}" | grep -qi 'sbatch: command not found'; then
    echo "Retrying sbatch via login shell (PATH from /etc/profile)..." >&2
    set +e
    JOB_LINE=$(
      _hpc_ssh "${HPC_USER}@${HPC_HOST}" \
        "bash -lc \"${_blc_prefix}cd \\\"${HPC_DIR}\\\" && chmod +x \\\"${SLURM_SCRIPT}\\\" && sbatch --parsable --export=\\\"${EXPORT_VARS}\\\" ${SBATCH_EXTRA} \\\"${SLURM_SCRIPT}\\\"\"" 2>&1
    )
    _ssh_rc=$?
    set -e
    echo "${JOB_LINE}" >&2
    JOB_ID=$(echo "${JOB_LINE}" | head -1 | cut -d';' -f1 | tr -d '[:space:]')
    if [ "${_ssh_rc}" -ne 0 ] || ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
      set +e
      JOB_LINE=$(
        _hpc_ssh "${HPC_USER}@${HPC_HOST}" \
          "bash -lc \"${_blc_prefix}cd \\\"${HPC_DIR}\\\" && chmod +x \\\"${SLURM_SCRIPT}\\\" && sbatch --export=\\\"${EXPORT_VARS}\\\" ${SBATCH_EXTRA} \\\"${SLURM_SCRIPT}\\\"\"" 2>&1
      )
      _ssh_rc=$?
      set -e
      echo "${JOB_LINE}" >&2
      JOB_ID=$(echo "${JOB_LINE}" | grep -oE '[0-9]+' | head -1)
    fi
  else
    JOB_ID=$(echo "${JOB_LINE}" | grep -oE '[0-9]+' | head -1)
  fi

  if [ "${_ssh_rc}" -ne 0 ]; then
    echo "" >&2
    echo "ERROR: ssh or sbatch failed (exit ${_ssh_rc}). See output above." >&2
    echo "  If you saw 'sbatch: command not found', on the login node run:  which sbatch" >&2
    echo "  Then:  export HPC_SBATCH=/full/path/to/sbatch   (or extend HPC_SLURM_PATH in hpc_defaults_from_local.sh)" >&2
    return 1
  fi
  if ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse JOB_ID from sbatch output." >&2
    return 1
  fi
  HPC_LAST_SSH_JOB_ID="${JOB_ID}"
  return 0
}

# mech_distill *_from_local.sh: non-interactive ssh often lacks sbatch on PATH; set -e would exit
# before printing "sbatch: command not found". On success sets HPC_LAST_SSH_JOB_ID; returns 0/1.
# Usage: hpc_ssh_sbatch_mech_distill "${HPC_DIR}" "${SLURM_SCRIPT}" "${EXPORT_VARS}" "${LAMBDA_MECH_LIST_CSV}"
_hpc_co2_load_project_env() {
  if [ "${ECHOHATE_SKIP_PROJECT_ENV:-0}" = "1" ]; then
    return 0
  fi
  local _js
  _js="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local _root="${ECHOHATE_ROOT:-$(cd "${_js}/.." && pwd)}"
  if [ -f "${_root}/config/echohate_project_config.json" ]; then
    ECHOHATE_ROOT="${_root}"
    # shellcheck source=/dev/null
    . "${_js}/echohate_project_env.sh"
  fi
}
_hpc_co2_load_project_env

hpc_ssh_sbatch_mech_distill() {
  local HPC_DIR="$1"
  local SLURM_SCRIPT="$2"
  local EXPORT_VARS="$3"
  EXPORT_VARS="${EXPORT_VARS}$(echohate_hpc_cuda_export_suffix)"
  local LAMBDA_MECH_LIST_CSV="$4"
  local _SLURM_ENV JOB_LINE _ssh_rc JOB_ID

  HPC_LAST_SSH_JOB_ID=""
  _SLURM_ENV="$(hpc_remote_slurm_env)"
  set +e
  JOB_LINE=$(
    _hpc_ssh "${HPC_USER}@${HPC_HOST}" \
      "${_SLURM_ENV}; cd \"${HPC_DIR}\" && chmod +x \"${SLURM_SCRIPT}\" && export LAMBDA_MECH_LIST_CSV='${LAMBDA_MECH_LIST_CSV}' && \"${HPC_SBATCH}\" --parsable --export=\"${EXPORT_VARS}\" \"${SLURM_SCRIPT}\"" 2>&1
  )
  _ssh_rc=$?
  set -e
  echo "${JOB_LINE}" >&2

  JOB_ID=$(echo "${JOB_LINE}" | head -1 | cut -d';' -f1 | tr -d '[:space:]')
  if [ "${_ssh_rc}" -eq 0 ] && [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    HPC_LAST_SSH_JOB_ID="${JOB_ID}"
    return 0
  elif echo "${JOB_LINE}" | grep -qiE 'unrecognised option|unrecognized option|invalid option.*parsable'; then
    echo "Retrying sbatch without --parsable (older Slurm)..." >&2
    set +e
    JOB_LINE=$(
      _hpc_ssh "${HPC_USER}@${HPC_HOST}" \
        "${_SLURM_ENV}; cd \"${HPC_DIR}\" && chmod +x \"${SLURM_SCRIPT}\" && export LAMBDA_MECH_LIST_CSV='${LAMBDA_MECH_LIST_CSV}' && \"${HPC_SBATCH}\" --export=\"${EXPORT_VARS}\" \"${SLURM_SCRIPT}\"" 2>&1
    )
    _ssh_rc=$?
    set -e
    echo "${JOB_LINE}" >&2
    JOB_ID=$(echo "${JOB_LINE}" | grep -oE '[0-9]+' | head -1)
  elif echo "${JOB_LINE}" | grep -qi 'sbatch: command not found'; then
    echo "Retrying sbatch via login shell (PATH from /etc/profile)..." >&2
    set +e
    JOB_LINE=$(
      _hpc_ssh "${HPC_USER}@${HPC_HOST}" \
        "bash -lc \"cd \\\"${HPC_DIR}\\\" && chmod +x \\\"${SLURM_SCRIPT}\\\" && export LAMBDA_MECH_LIST_CSV='${LAMBDA_MECH_LIST_CSV}' && sbatch --parsable --export=\\\"${EXPORT_VARS}\\\" \\\"${SLURM_SCRIPT}\\\"\"" 2>&1
    )
    _ssh_rc=$?
    set -e
    echo "${JOB_LINE}" >&2
    JOB_ID=$(echo "${JOB_LINE}" | head -1 | cut -d';' -f1 | tr -d '[:space:]')
    if [ "${_ssh_rc}" -ne 0 ] || ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
      set +e
      JOB_LINE=$(
        _hpc_ssh "${HPC_USER}@${HPC_HOST}" \
          "bash -lc \"cd \\\"${HPC_DIR}\\\" && chmod +x \\\"${SLURM_SCRIPT}\\\" && export LAMBDA_MECH_LIST_CSV='${LAMBDA_MECH_LIST_CSV}' && sbatch --export=\\\"${EXPORT_VARS}\\\" \\\"${SLURM_SCRIPT}\\\"\"" 2>&1
      )
      _ssh_rc=$?
      set -e
      echo "${JOB_LINE}" >&2
      JOB_ID=$(echo "${JOB_LINE}" | grep -oE '[0-9]+' | head -1)
    fi
  else
    JOB_ID=$(echo "${JOB_LINE}" | grep -oE '[0-9]+' | head -1)
  fi

  if [ "${_ssh_rc}" -ne 0 ]; then
    echo "" >&2
    echo "ERROR: ssh or sbatch failed (exit ${_ssh_rc}). See output above." >&2
    echo "  If you saw 'sbatch: command not found', on the login node run:  which sbatch" >&2
    echo "  Then:  export HPC_SBATCH=/full/path/to/sbatch   (or extend HPC_SLURM_PATH in hpc_defaults_from_local.sh)" >&2
    return 1
  fi
  if ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse JOB_ID from sbatch output." >&2
    return 1
  fi
  HPC_LAST_SSH_JOB_ID="${JOB_ID}"
  return 0
}

# HF token helpers (secrets/hf_token); call echohate_export_hf_token_from_secrets after ECHOHATE_ROOT is known.
_hpc_defaults_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
[ -f "${_hpc_defaults_dir}/echohate_hf_token_env.sh" ] && source "${_hpc_defaults_dir}/echohate_hf_token_env.sh"
# shellcheck source=/dev/null
[ -f "${_hpc_defaults_dir}/echohate_hpc_password_env.sh" ] && source "${_hpc_defaults_dir}/echohate_hpc_password_env.sh"
echohate_export_hpc_password_from_secrets "$(cd "${_hpc_defaults_dir}/.." && pwd)" 2>/dev/null || true
