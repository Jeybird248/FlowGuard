#!/bin/bash
# Helper: run a Python module inside the FlowGuard apptainer image with the
# repo bind-mounted at /work.
#
# Usage: bash src/scripts/_apptainer_exec.sh python -m flowguard.scripts.extract_features ...

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
container_name="${FLOWGUARD_CONTAINER_NAME:-flowguard}"
containers_dir="${FLOWGUARD_CONTAINERS_DIR:-${repo_root}/containers}"
sif="${containers_dir}/${container_name}.sif"

if [ ! -f "${sif}" ]; then
  echo "[apptainer_exec] image not found: ${sif}" >&2
  echo "Build it first: bash containers/build_container.sh ${container_name}" >&2
  exit 4
fi

binds=("${repo_root}:/work")
[ -n "${HF_HOME:-}" ] && binds+=("${HF_HOME}:${HF_HOME}")
[ -n "${FLOWGUARD_RESULTS_DIR:-}" ] && binds+=("${FLOWGUARD_RESULTS_DIR}:${FLOWGUARD_RESULTS_DIR}")
[ -n "${FLOWGUARD_FEATURES_DIR:-}" ] && binds+=("${FLOWGUARD_FEATURES_DIR}:${FLOWGUARD_FEATURES_DIR}")
[ -n "${FLOWGUARD_DATA_DIR:-}" ] && binds+=("${FLOWGUARD_DATA_DIR}:${FLOWGUARD_DATA_DIR}")

bind_args=()
for b in "${binds[@]}"; do
  bind_args+=("--bind" "${b}")
done

export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

exec apptainer exec --nv "${bind_args[@]}" --pwd /work "${sif}" "$@"
