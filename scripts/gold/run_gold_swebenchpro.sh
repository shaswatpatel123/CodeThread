#!/bin/bash

#SBATCH --job-name=sp_gold
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/spp9399/output_logs/codingAgent_output/gold_batch_%A_%a.out
#SBATCH --mail-user=spp9399@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --account=torch_pr_221_courant
#SBATCH --gres=gpu:l40s:1

module purge


export MSWEA_SINGULARITY_EXECUTABLE=/share/apps/apptainer/1.4.5/bin/apptainer
export APPTAINER_CACHEDIR=/scratch/spp9399/.apptainer/cache
export TMPDIR=/scratch/spp9399/.apptainer/tmp
export APPTAINER_TMPDIR=/scratch/spp9399/.apptainer/tmp
export APPTAINER_WORKDIR=/scratch/spp9399/.apptainer/work
export APPTAINER_CONFIGDIR=/scratch/spp9399/.apptainer

GPUTRICK_LOG=/scratch/spp9399/output_logs/codingAgent_output/gpuTrick_${TIMESTAMP}.out
bash -c "/scratch/spp9399/env/retrieval_heads/run.sh bash ./runGPUTrick.sh">>"$GPUTRICK_LOG" 2>&1 &
GPUTRICK_PID=$!

# Clear the tmp files
echo "Clear the tmp files"
find /scratch/spp9399/.apptainer/tmp/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer/cache/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"
find /scratch/spp9399/.apptainer/work/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P 8 -I {} rm -rf "{}"

module load anaconda3/2025.06
source activate /scratch/spp9399/MaintainableCoder/penv

export CUSTOM_DATA_PATH="/scratch/spp9399/MaintainableCoder/minisweagent/experiments/data/swebenchpr_w_problem_statement.csv"
mini-extra swebenchpro \
    --subset swebenchpro \
    --split test \
    --output ../../results/gold_swebenchpro  \
    --config ../../../src/minisweagent/config/extra/swebench_qwen_swebenchpro.yaml \
    --workers 16 \
    --environment-class singularity \
    --gold \
    --path-local-images /scratch/spp9399/dockerImages/swebenchpro/ \
    --run-only-eval \
    --redo-existing \
    --multilingual \
    --scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/run_scripts/" \
    --docker-scripts-dir "/scratch/spp9399/MaintainableCoder/minisweagent/src/minisweagent/evaluation/" \
    --task-column-name "PR_1_problem_statement" \
    --user-custom
