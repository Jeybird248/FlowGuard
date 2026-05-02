#!/bin/bash
# Source this file before submitting jobs:
#   source env.sh
#
# Do not commit real credentials — this file is checked in as a template.

export HF_TOKEN="${HF_TOKEN:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# Shared caches. /fast has no flock support; the container ships
# soft_file_locks so HF / torch hub still operate.
user_name="${USER:-$(id -un)}"
export HF_HOME="${HF_HOME:-/fast/${user_name}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

# FlowGuard-specific paths.
export FLOWGUARD_RESULTS_DIR="${FLOWGUARD_RESULTS_DIR:-/fast/${user_name}/flowguard_results}"
export FLOWGUARD_FEATURES_DIR="${FLOWGUARD_FEATURES_DIR:-/fast/${user_name}/flowguard_features}"
export FLOWGUARD_DATA_DIR="${FLOWGUARD_DATA_DIR:-/fast/${user_name}/flowguard_data}"
export FLOWGUARD_CONTAINERS_DIR="${FLOWGUARD_CONTAINERS_DIR:-/fast/${user_name}/flowguard_containers}"
export FLOWGUARD_CONTAINER_NAME="${FLOWGUARD_CONTAINER_NAME:-flowguard}"

mkdir -p "${FLOWGUARD_RESULTS_DIR}" "${FLOWGUARD_FEATURES_DIR}" \
         "${FLOWGUARD_DATA_DIR}" "${FLOWGUARD_CONTAINERS_DIR}" \
         "${HF_HUB_CACHE}" 2>/dev/null || true

export FLOWGUARD_JOB_SCHEDULER="${FLOWGUARD_JOB_SCHEDULER:-htcondor}"
