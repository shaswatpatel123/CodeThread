#!/bin/bash

export MSWEA_SINGULARITY_EXECUTABLE=/share/apps/apptainer/1.4.5/bin/apptainer
export APPTAINER_CACHEDIR=/scratch/spp9399/.apptainer2/cache
export TMPDIR=/scratch/spp9399/.apptainer2/tmp
export APPTAINER_TMPDIR=/scratch/spp9399/.apptainer2/tmp
export APPTAINER_WORKDIR=/scratch/spp9399/.apptainer2/work
export APPTAINER_CONFIGDIR=/scratch/spp9399/.apptainer2

# Clear the tmp files 
find /scratch/spp9399/.apptainer2/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"

export CUSTOM_DATA_PATH="../dataset/data/swebench_multilingual.csv"
mini-extra swebench \
    --subset multilingual \
    --split test \
    --output ./results/glm47_first_multilingual \
    --config ../src/minisweagent/config/extra/swebench_glm47.yaml \
    --workers 16 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebench_multilingual \
    --task-column-name "PR_1_problem_statement" \
    --multilingual \
    --user-custom

# Clear the tmp files
find /scratch/spp9399/.apptainer2/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"

export CUSTOM_DATA_PATH="../dataset/data/swebench_multilingual.csv"
mini-extra swebench \
    --subset multilingual \
    --split test \
    --output ./results/glm47_first_multilingual \
    --config ../src/minisweagent/config/extra/swebench_glm47.yaml \
    --workers 16 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebench_multilingual \
    --task-column-name "PR_1_problem_statement" \
    --multilingual \
    --user-custom \
    --redo-existing \
    --run-only-eval

FIRST_PR_RESULT_PATH=./results/glm47_first_multilingual

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

echo "Clear the tmp files ..."
# Clear the tmp files
find /scratch/spp9399/.apptainer2/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"

export CUSTOM_DATA_PATH="../dataset/data/swebench_multilingual.csv"
mini-extra swebench \
    --subset multilingual \
    --output ./results/glm47_second_multilingual  \
    --config ../src/minisweagent/config/extra/swebench_glm47.yaml  \
    --split test \
    --workers 16 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebench_multilingual \
    --filter-ids "$FILTER_IDS" \
    --init-patch-map $PATCH_MAP_FILE \
    --multilingual \
    --user-custom

echo "Clear the tmp files ..."
# Clear the tmp files
find /scratch/spp9399/.apptainer2/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"

export CUSTOM_DATA_PATH="../dataset/data/swebench_multilingual.csv"
mini-extra swebench \
    --subset multilingual \
    --output ./results/glm47_second_multilingual  \
    --config ../src/minisweagent/config/extra/swebench_glm47.yaml  \
    --split test \
    --workers 16 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebench_multilingual \
    --filter-ids "$FILTER_IDS" \
    --init-patch-map $PATCH_MAP_FILE \
    --multilingual \
    --user-custom \
    --redo-existing \
    --run-only-eval

echo "Clear the tmp files ..."
# Clear the tmp files
find /scratch/spp9399/.apptainer2/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"


mini-extra swebench \
    --subset multilingual \
    --split test \
    --output ./results/GLM47_Multi \
    --config ../src/minisweagent/config/extra/swebench_glm47.yaml \
    --workers 8 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebench_multilingual \
    --multilingual \
    --filter-ids "$FILTER_IDS" \
    --multilingual


echo "Clear the tmp files ..."
# Clear the tmp files
find /scratch/spp9399/.apptainer2/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer2/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"


mini-extra swebench \
    --subset multilingual \
    --split test \
    --output ./results/GLM47_Multi \
    --config ../src/minisweagent/config/extra/swebench_glm47.yaml \
    --workers 16 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebench_multilingual \
    --multilingual \
    --filter-ids "$FILTER_IDS" \
    --multilingual \
    --redo-existing \
    --run-only-eval