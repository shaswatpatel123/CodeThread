#!/bin/bash

module load anaconda3/2025.06
source activate /scratch/spp9399/MaintainableCoder/penv

export MSWEA_SINGULARITY_EXECUTABLE=/share/apps/apptainer/1.4.5/bin/apptainer
export APPTAINER_CACHEDIR=/scratch/spp9399/.apptainer3/cache
export TMPDIR=/scratch/spp9399/.apptainer3/tmp
export APPTAINER_TMPDIR=/scratch/spp9399/.apptainer3/tmp
export APPTAINER_WORKDIR=/scratch/spp9399/.apptainer3/work
export APPTAINER_CONFIGDIR=/scratch/spp9399/.apptainer3

# Function to clean apptainer directories
clean_apptainer() {
    echo "Removing tmp cache work"
    find /scratch/spp9399/.apptainer3/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 4 -I {} rm -rf "{}"
    find /scratch/spp9399/.apptainer3/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 4 -I {} rm -rf "{}"
    find /scratch/spp9399/.apptainer3/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 4 -I {} rm -rf "{}"
}

# Clean before starting
clean_apptainer

SECOND_PR_OUTPUT_FOLDER=../results/synthetic_chains_glm47_second_easy_swebenchpro
FIRST_PR_OUTPUT_FOLDER=../results/synthetic_glm47_first_easy_swebenchpro
GOLD_PR_OUTPUT_FOLDER=../results/gold_swebenchpro
SWEBENCH_PR_OUTPUT_FOLDER=../results/GLM47_swebenchpro

OUTPUT_FOLDER=../results/maintainability_result_swebench/glm47_swebenchpro

instances=$(
find "$SECOND_PR_OUTPUT_FOLDER" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
| sort \
| paste -sd '|' -
)

export CUSTOM_DATA_PATH="/scratch/spp9399/MaintainableCoder/minisweagent/experiments/data/swebenchpr_w_problem_statement.csv"
mini-extra maintainability-fast-swebenchpro \
	--subset swebenchpro \
	--split test \
	--output ${OUTPUT_FOLDER} \
	--config /scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/config/extra/swebench_glm47_swebenchpro_nonlocal.yaml \
	--workers 8 \
	--task-column-name "PR_1_problem_statement" \
	--environment-class maintainability_fast \
	--path-local-images /scratch/spp9399/dockerImages/swebenchpro \
	--gold-patch-path ${GOLD_PR_OUTPUT_FOLDER} \
        --swebench-patch-path ${SWEBENCH_PR_OUTPUT_FOLDER} \
	--first-pr-patch-path ${FIRST_PR_OUTPUT_FOLDER} \
	--second-pr-patch-path ${SECOND_PR_OUTPUT_FOLDER} \
	--init-patch-map ${FIRST_PR_OUTPUT_FOLDER}/secondPRMapper.json \
	--multilingual \
	--user-custom \
	--redo-existing

# Clean after each iteration
clean_apptainer