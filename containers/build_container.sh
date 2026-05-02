#!/bin/bash
# Build a FlowGuard apptainer image from containers/<name>.def.
#
# Usage:
#   bash containers/build_container.sh flowguard
#   bash containers/build_container.sh flowguard_a100
#
# Output goes to ${FLOWGUARD_CONTAINERS_DIR:-containers}/<name>.sif. Build logs
# are tee'd to <containers_dir>/build_logs/<name>_build_<timestamp>.log and a
# rolling latest symlink at <containers_dir>/build_logs/<name>_build.log.

set -euo pipefail

container="${1:-}"
def_override="${2:-}"
if [ -z "${container}" ]; then
  echo "Usage: $0 <container-name> [def-file]" >&2
  exit 2
fi

export FLOWGUARD_CONTAINERS_DIR=${FLOWGUARD_CONTAINERS_DIR:-containers}
export APPTAINER_BIND=""

mkdir -p "${FLOWGUARD_CONTAINERS_DIR}"
out_file="${FLOWGUARD_CONTAINERS_DIR}/${container}.sif"
def_file="${def_override:-containers/${container}.def}"

if [[ "${FLOWGUARD_CONTAINERS_DIR}" = /* ]]; then
  log_root="${FLOWGUARD_CONTAINERS_DIR}/build_logs"
else
  log_root="$(pwd)/${FLOWGUARD_CONTAINERS_DIR}/build_logs"
fi
mkdir -p "${log_root}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${log_root}/${container}_build_${timestamp}.log"
latest_link="${log_root}/${container}_build.log"

exec > >(tee -a "${log_file}") 2>&1
ln -sfn "$(basename "${log_file}")" "${latest_link}" 2>/dev/null || true
echo "[build_container] build_log=${log_file}"

_on_exit() {
  local rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo "[build_container] FAILED (exit=${rc}). See log: ${log_file}" >&2
    tail -n 80 "${log_file}" >&2 || true
  else
    echo "[build_container] SUCCESS. Log: ${log_file}"
  fi
}
trap _on_exit EXIT

if [ ! -f "${def_file}" ]; then
  echo "[build_container] ERROR: definition file not found: ${def_file}" >&2
  exit 2
fi

user_name="${USER:-$(id -un 2>/dev/null || echo user)}"

# Apptainer needs a tmpdir that supports flock; /home does, /fast does not.
if [ -z "${APPTAINER_TMPDIR:-}" ]; then
  export APPTAINER_TMPDIR="${HOME%/}/apptainer_tmp"
fi
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/fast/${user_name}/apptainer_cache}"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

# Skip xattrs and cap mksquashfs memory — mirrors the InferenceBench setup so
# the build succeeds on shared login nodes.
export APPTAINER_SQUASHFS_OPTS="${APPTAINER_SQUASHFS_OPTS:--no-xattrs -processors 1 -mem 1024M -comp gzip}"
export SINGULARITY_SQUASHFS_OPTS="${SINGULARITY_SQUASHFS_OPTS:-${APPTAINER_SQUASHFS_OPTS}}"
export APPTAINER_MKSQUASHFS_PROCS="${APPTAINER_MKSQUASHFS_PROCS:-1}"
export APPTAINER_MKSQUASHFS_MEM="${APPTAINER_MKSQUASHFS_MEM:-1024M}"
export SINGULARITY_MKSQUASHFS_PROCS="${SINGULARITY_MKSQUASHFS_PROCS:-${APPTAINER_MKSQUASHFS_PROCS}}"
export SINGULARITY_MKSQUASHFS_MEM="${SINGULARITY_MKSQUASHFS_MEM:-${APPTAINER_MKSQUASHFS_MEM}}"

fakeroot_flag=""
if [ "$(id -u)" -ne 0 ]; then
  fakeroot_flag="--fakeroot"
fi

build_cmd=(apptainer build)
[ -n "${fakeroot_flag}" ] && build_cmd+=("${fakeroot_flag}")
if apptainer build --help 2>/dev/null | grep -q -- '--mksquashfs-args'; then
  build_cmd+=(--mksquashfs-args "${APPTAINER_SQUASHFS_OPTS}")
fi
build_cmd+=("${out_file}" "${def_file}")
echo "[build_container] command=${build_cmd[*]}"

"${build_cmd[@]}"

if [ ! -f "${out_file}" ]; then
  echo "[build_container] ERROR: output file missing: ${out_file}" >&2
  exit 3
fi
echo "[build_container] built: $(realpath "${out_file}")"
ls -lh "${out_file}"
