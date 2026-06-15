#!/bin/bash
#
# CodeThread dataset generation pipeline
# =====================================
# Two stages:
#   1. data_gen_patches.py  (CPU, no LLM) - clone repos, extract modified
#      functions from gold patches, build PR0 stub patches -> intermediate parquet
#   2. data_gen_problems.py (GPU, vLLM)   - generate problem statements from the
#      intermediate parquet -> final parquet
#
# Run from this directory:  bash run_problem.sh
# Edit the variables below to match your setup.

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR="./data/"

# Datasets to process (space-separated). Options: multilingual pro featurebench featbench
# Use "all" to process and concatenate every supported dataset.
DATASETS="all"

# Where repos are cloned for stage 1 (add --skip_clone below to reuse existing clones).
REPO_BASE_PATH="cloned_repos"
CLONE_WORKERS=4

# Stage 2 (vLLM) settings.
MODEL_NAME="zai-org/GLM-4.7-Flash"   # HF name or local path to the model
NUM_GPUS=1                            # tensor_parallel_size
BATCH_SIZE=4

# Optional launcher that brings up the vLLM environment before stage 2
# (e.g. a cluster wrapper script). Leave empty to run python3 directly.
VLLM_LAUNCHER="${VLLM_LAUNCHER:-}"

# ---------------------------------------------------------------------------
PATCHES_PARQUET="${OUTPUT_DIR}/patches.parquet"
FINAL_PARQUET="${OUTPUT_DIR}/dataset_with_problem_statements.parquet"

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Stage 1: patch processing (CPU)
# ---------------------------------------------------------------------------
echo "=== Stage 1: data_gen_patches.py -> ${PATCHES_PARQUET} ==="
python3 data_gen_patches.py \
    --datasets ${DATASETS} \
    --output "${PATCHES_PARQUET}" \
    --repo_base_path "${REPO_BASE_PATH}" \
    --clone_workers "${CLONE_WORKERS}"
    # --skip_clone           # uncomment to reuse repos already at REPO_BASE_PATH
    # --num_instances 10     # uncomment to process only the first N per dataset

# ---------------------------------------------------------------------------
# Stage 2: problem statement generation (GPU / vLLM)
# ---------------------------------------------------------------------------
echo "=== Stage 2: data_gen_problems.py -> ${FINAL_PARQUET} ==="
${VLLM_LAUNCHER} python3 data_gen_problems.py \
    --input "${PATCHES_PARQUET}" \
    --output "${FINAL_PARQUET}" \
    --model_name "${MODEL_NAME}" \
    --num_gpus "${NUM_GPUS}" \
    --batch_size "${BATCH_SIZE}"

echo "=== Done. Final dataset: ${FINAL_PARQUET} ==="
