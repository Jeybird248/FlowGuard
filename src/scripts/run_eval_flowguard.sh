#!/bin/bash
# Condor entrypoint: evaluate FlowGuard on (model, benchmark) at one seed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

model="${1:?model required}"
benchmark="${2:?benchmark required}"
seed="${3:-248}"

detector_path="${FLOWGUARD_FEATURES_DIR}/${model}/detector_seed${seed}.pkl"

bash src/scripts/_apptainer_exec.sh python -m flowguard.scripts.eval_flowguard \
    --model "${model}" \
    --benchmark "${benchmark}" \
    --seed "${seed}" \
    --detector-path "${detector_path}"
