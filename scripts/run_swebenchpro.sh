#!/bin/bash

export MSWEA_SINGULARITY_EXECUTABLE=/share/apps/apptainer/1.4.5/bin/apptainer
export APPTAINER_CACHEDIR=/scratch/spp9399/.apptainer3/cache
export TMPDIR=/scratch/spp9399/.apptainer3/tmp
export APPTAINER_TMPDIR=/scratch/spp9399/.apptainer3/tmp
export APPTAINER_WORKDIR=/scratch/spp9399/.apptainer3/work
export APPTAINER_CONFIGDIR=/scratch/spp9399/.apptainer3

# Clear the tmp files 
echo "Clear the tmp"
find /scratch/spp9399/.apptainer3/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer3/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer3/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"


export CUSTOM_DATA_PATH="/scratch/spp9399/MaintainableCoder/minisweagent/experiments/data/swebenchpr_w_problem_statement.csv"
mini-extra swebenchpro \
    --subset swebenchpro \
    --split test \
    --output ./results/synthetic_glm47_first_swebenchpro \
    --config ../src/minisweagent/config/extra/swebench_glm47_swebenchpro.yaml \
    --workers 12 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebenchpro \
    --task-column-name "PR_1_problem_statement" \
    --scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/run_scripts/" \
    --docker-scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/" \
    --multilingual \
    --user-custom \

echo "Clear the tmp files ..."
# Clear the tmp files
find /scratch/spp9399/.apptainer3/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer3/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer3/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"


export CUSTOM_DATA_PATH="/scratch/spp9399/MaintainableCoder/minisweagent/experiments/data/swebenchpr_w_problem_statement.csv"
mini-extra swebenchpro \
    --subset swebenchpro \
    --split test \
    --output ./results/synthetic_glm47_first_swebenchpro \
    --config ../src/minisweagent/config/extra/swebench_glm47_swebenchpro.yaml \
    --workers 16 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebenchpro \
    --task-column-name "PR_1_problem_statement" \
    --scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/run_scripts/" \
    --docker-scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/" \
    --multilingual \
    --user-custom \
    --redo-existing \
    --run-only-eval

FIRST_PR_RESULT_PATH=./results/synthetic_glm47_first_swebenchpro

# ----------------------------------------------------------------------
# 1. Generate synthetic report using synthetic_chains.py
# ----------------------------------------------------------------------
echo "Generating synthetic_report.json"
python3 /scratch/spp9399/MaintainableCoder/minisweagent/experiments/scripts/qwen/run_scripts/utils/synthetic_chains.py "${FIRST_PR_RESULT_PATH}"
echo "Generated synthetic_report.json"

# ----------------------------------------------------------------------
# 1. Generate filter-ids using get_instance_ids.py
# ----------------------------------------------------------------------
echo "Generating FILTER_IDS ..."
FILTER_IDS=$(python3 /scratch/spp9399/MaintainableCoder/minisweagent/experiments/scripts/qwen/run_scripts/utils/get_instance_ids.py "${FIRST_PR_RESULT_PATH}")
echo "FILTER_IDS: $FILTER_IDS"

# ----------------------------------------------------------------------
# 2. Generate init patch map JSON using generate_2ndPRJson.py
# ----------------------------------------------------------------------
PATCH_MAP_FILE="${FIRST_PR_RESULT_PATH}/secondPRMapper.json"

echo "Generating init-patch-map JSON ..."
python3 /scratch/spp9399/MaintainableCoder/minisweagent/experiments/scripts/qwen/run_scripts/utils/generate_2ndPRJson.py "${FIRST_PR_RESULT_PATH}"
echo "Patch map saved to: $PATCH_MAP_FILE"
# ----------------------------------------------------------------------
# 3. Run mini-extra
# ----------------------------------------------------------------------


echo "Clear the tmp files ..."
# Clear the tmp files
find /scratch/spp9399/.apptainer3/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer3/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer3/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"

export CUSTOM_DATA_PATH="/scratch/spp9399/MaintainableCoder/minisweagent/experiments/data/swebenchpr_w_problem_statement.csv"
mini-extra swebenchpro \
    --subset swebenchpro \
    --split test \
    --multilingual \
    --workers 16 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebenchpro \
    --filter-ids "$FILTER_IDS" \
    --init-patch-map $PATCH_MAP_FILE \
    --output ./results/synthetic_chains_glm47_second_swebenchpro \
    --config ../src/minisweagent/config/extra/swebench_glm47_swebenchpro.yaml \
    --scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/run_scripts/" \
    --docker-scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/" \
    --user-custom


echo "Clear the tmp files ..."
# Clear the tmp files
find /scratch/spp9399/.apptainer3/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer3/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer3/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"


export CUSTOM_DATA_PATH="/scratch/spp9399/MaintainableCoder/minisweagent/experiments/data/swebenchpr_w_problem_statement.csv"
mini-extra swebenchpro \
    --subset swebenchpro \
    --split test \
    --multilingual \
    --workers 16 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebenchpro \
    --init-patch-map $PATCH_MAP_FILE \
    --output ./results/synthetic_chains_glm47_second_swebenchpro \
    --config ../src/minisweagent/config/extra/swebench_glm47_swebenchpro.yaml \
    --scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/run_scripts/" \
    --docker-scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/" \
    --user-custom \
    --redo-existing \
    --run-only-eval


echo "Clear the tmp files ..."
# Clear the tmp files
find /scratch/spp9399/.apptainer3/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer3/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer3/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"

mini-extra swebenchpro \
    --subset swebenchpro \
    --split test \
    --output ./results/GLM47_swebenchpro \
    --config ../src/minisweagent/config/extra/swebench_glm47_swebenchpro.yaml \
    --workers 16 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebenchpro \
    --filter-ids "$FILTER_IDS" \
    --scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/run_scripts/" \
    --docker-scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/" \
    --multilingual

echo "Clear the tmp files ..."
# Clear the tmp files
find /scratch/spp9399/.apptainer3/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer3/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer3/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"

mini-extra swebenchpro \
    --subset swebenchpro \
    --split test \
    --output ./results/GLM47_swebenchpro \
    --config ../src/minisweagent/config/extra/swebench_glm47_swebenchpro.yaml \
    --workers 10 \
    --environment-class singularity \
    --path-local-images /scratch/spp9399/dockerImages/swebenchpro \
    --filter-ids "$FILTER_IDS" \
    --scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/run_scripts/" \
    --docker-scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/" \
    --multilingual \
    --redo-existing \
    --run-only-eval
