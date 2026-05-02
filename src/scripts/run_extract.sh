#!/bin/bash
# Condor entrypoint: extract FlowVectors for a single (model, split) combination.
# Arguments are positional and matched to extract_features.sub:
#   $1 model     (e.g. llava-1.5-7b)
#   $2 split     (e.g. vqav2_train, mmsb)
#   $3 n         (cap; pass -1 for no cap)
#   $4 seed
#   $5 cluster_id (unused; condor injects)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

model="${1:?model required}"
split="${2:?split required}"
n="${3:-10000}"
seed="${4:-248}"

cmd=(python -m flowguard.scripts.extract_features
     --model "${model}"
     --split "${split}"
     --seed  "${seed}")
if [ "${n}" != "-1" ]; then
  cmd+=("--n" "${n}")
fi

bash src/scripts/_apptainer_exec.sh "${cmd[@]}"
