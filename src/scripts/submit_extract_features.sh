#!/bin/bash
# Submit FlowVector extraction jobs to HTCondor.
#
#   bash src/scripts/submit_extract_features.sh \
#       --model llava-1.5-7b --split vqav2_train --n 10000 --seed 248
#
# Adversarial inputs come from the dataset loader directly (e.g. ``mmsb``,
# ``vlsafe``, ``vlsu_unsafe``); FlowGuard does not generate attacks itself.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"

MODEL=""
SPLIT=""
N=""
SEED=248
BID=1000
DRY_RUN=0
SUBFILE="condor/extract_features.sub"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   shift; MODEL="$1" ;;
    --split)   shift; SPLIT="$1" ;;
    --n)       shift; N="$1" ;;
    --seed)    shift; SEED="$1" ;;
    --bid)     shift; BID="$1" ;;
    --sub)     shift; SUBFILE="$1" ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done
[ -z "${MODEL}" ] && { echo "--model required"; exit 1; }
[ -z "${SPLIT}" ] && { echo "--split required"; exit 1; }
[ -z "${N}" ] && N="-1"

log_dir="${FLOWGUARD_RESULTS_DIR}/condor_logs/extract"
mkdir -p "${log_dir}"
export FLOWGUARD_LOG_DIR="${log_dir}"

submit_args=(
  "${SUBFILE}"
  -append "model=${MODEL}"
  -append "split=${SPLIT}"
  -append "n=${N}"
  -append "seed=${SEED}"
  -append "FLOWGUARD_LOG_DIR=${log_dir}"
)

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "would submit: condor_submit_bid ${BID} ${submit_args[*]}"
  exit 0
fi
condor_submit_bid "${BID}" "${submit_args[@]}"
