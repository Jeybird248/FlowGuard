#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

model="${1:?model required}"
seed="${2:-248}"
contamination="${3:-auto}"

bash src/scripts/_apptainer_exec.sh python -m flowguard.scripts.train_detector \
    --model "${model}" \
    --seed "${seed}" \
    --contamination "${contamination}"
