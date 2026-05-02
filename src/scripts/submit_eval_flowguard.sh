#!/bin/bash
# Submit FlowGuard evaluation jobs across (benchmark, seed) combinations.
#
#   bash src/scripts/submit_eval_flowguard.sh \
#       --model llava-1.5-7b \
#       --benchmarks mmsb,vlsafe,vlsu_unsafe,vqav2_val,mossbench,vizwiz_val \
#       --seeds 21,248,999

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"

MODEL=""
BENCHMARKS="mmsb,vlsafe,vlsu_unsafe,vqav2_val,mossbench,vizwiz_val"
SEEDS="21,248,999"
BID=1000
SUBFILE="condor/eval_flowguard.sub"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       shift; MODEL="$1" ;;
    --benchmarks)  shift; BENCHMARKS="$1" ;;
    --seeds)       shift; SEEDS="$1" ;;
    --bid)         shift; BID="$1" ;;
    --sub)         shift; SUBFILE="$1" ;;
    --dry-run)     DRY_RUN=1 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done
[ -z "${MODEL}" ] && { echo "--model required"; exit 1; }

IFS=',' read -ra BENCHES <<< "${BENCHMARKS}"
IFS=',' read -ra SDS <<< "${SEEDS}"

log_dir="${FLOWGUARD_RESULTS_DIR}/condor_logs/eval"
mkdir -p "${log_dir}"
export FLOWGUARD_LOG_DIR="${log_dir}"

n_total=0
for bench in "${BENCHES[@]}"; do
  for seed in "${SDS[@]}"; do
    args=(
      "${SUBFILE}"
      -append "model=${MODEL}"
      -append "benchmark=${bench}"
      -append "seed=${seed}"
      -append "FLOWGUARD_LOG_DIR=${log_dir}"
    )
    if [ "${DRY_RUN}" -eq 1 ]; then
      echo "condor_submit_bid ${BID} ${args[*]}"
    else
      condor_submit_bid "${BID}" "${args[@]}"
    fi
    n_total=$((n_total + 1))
  done
done

echo "[submit_eval_flowguard] queued ${n_total} jobs"
