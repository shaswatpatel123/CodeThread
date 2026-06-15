#!/bin/bash

module purge

export MSWEA_SINGULARITY_EXECUTABLE=/share/apps/apptainer/1.4.5/bin/apptainer
export APPTAINER_CACHEDIR=/scratch/spp9399/.apptainer2/cache
export TMPDIR=/scratch/spp9399/.apptainer2/tmp
export APPTAINER_TMPDIR=/scratch/spp9399/.apptainer2/tmp
export APPTAINER_WORKDIR=/scratch/spp9399/.apptainer2/work
export APPTAINER_CONFIGDIR=/scratch/spp9399/.apptainer2

# Pre-run cleanup (no workers active, safe to clean everything)

echo "Clear the tmp files ..."
# Clear the tmp files
find /scratch/spp9399/.apptainer2/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"


export CUSTOM_DATA_PATH="../dataset/data/swebench_verified/swebench_verified.csv"
mini-extra swebench-old \
    --subset custom \
    --split train \
    --output ../../results/glm47_swebench \
    --config ../../../src/minisweagent/config/extra/swebench.yaml \
    --workers 16 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebench_verified/ \
    --task-column-name "problem_statement" \
    --use-corruption --is-sanity-run

# Clear the tmp files

echo "Clear the tmp files ..."
# Clear the tmp files
find /scratch/spp9399/.apptainer2/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"

export CUSTOM_DATA_PATH="../dataset/data/swebench_verified/swebench_verified.csv"
mini-extra swebench-old \
    --subset custom \
    --split train \
    --output ../../results/glm47_swebench \
    --config ../../../src/minisweagent/config/extra/swebench.yaml \
    --workers 32 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebench_verified/ \
    --task-column-name "problem_statement" \
    --redo-existing \
    --run-only-eval \
    --use-corruption --is-sanity-run 

FIRST_PR_RESULT_PATH=../../results/glm47_swebench

# ----------------------------------------------------------------------
# 1. Generate synthetic report using synthetic_chains.py
# ----------------------------------------------------------------------
echo "Generating synthetic_report.json"
python3 ./utils/synthetic_chains.py "${FIRST_PR_RESULT_PATH}"
echo "Generated synthetic_report.json"

# ----------------------------------------------------------------------
# 1. Generate filter-ids using get_instance_ids.py
# ----------------------------------------------------------------------
echo "Generating FILTER_IDS ..."
FILTER_IDS=$(python3 ./utils/get_instance_ids.py "${FIRST_PR_RESULT_PATH}")
echo "FILTER_IDS: $FILTER_IDS"

# ----------------------------------------------------------------------
# 2. Generate init patch map JSON using generate_2ndPRJson.py
# ----------------------------------------------------------------------
PATCH_MAP_FILE="${FIRST_PR_RESULT_PATH}/secondPRMapper.json"

echo "Generating init-patch-map JSON ..."
python3 ./utils/generate_2ndPRJson.py "${FIRST_PR_RESULT_PATH}"
echo "Patch map saved to: $PATCH_MAP_FILE"
# ----------------------------------------------------------------------
# 3. Run mini-extra
# ----------------------------------------------------------------------

# Follow run_swebenchpro.sh for 2nd PR and HA evaluation