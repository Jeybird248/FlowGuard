#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"

MODEL=""
SEED=248
CONTAMINATION="auto"
BID=1000
DRY_RUN=0
SUBFILE="condor/train_detector.sub"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)         shift; MODEL="$1" ;;
    --seed)          shift; SEED="$1" ;;
    --contamination) shift; CONTAMINATION="$1" ;;
    --bid)           shift; BID="$1" ;;
    --sub)           shift; SUBFILE="$1" ;;
    --dry-run)       DRY_RUN=1 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done
[ -z "${MODEL}" ] && { echo "--model required"; exit 1; }

log_dir="${FLOWGUARD_RESULTS_DIR}/condor_logs/train"
mkdir -p "${log_dir}"
export FLOWGUARD_LOG_DIR="${log_dir}"

args=(
  "${SUBFILE}"
  -append "model=${MODEL}"
  -append "seed=${SEED}"
  -append "contamination=${CONTAMINATION}"
  -append "FLOWGUARD_LOG_DIR=${log_dir}"
)

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "would submit: condor_submit_bid ${BID} ${args[*]}"
  exit 0
fi
condor_submit_bid "${BID}" "${args[@]}"
