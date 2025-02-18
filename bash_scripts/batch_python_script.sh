#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu
#SBATCH --mem=90G
#SBATCH --partition=gpulong
#SBATCH --output="./output_scripts/%j_script.out"

echo "conda env: $CONDA_PROMPT_MODIFIER"
echo "Start of run: $(date +%Y%m%d_%H:%M)"
echo "Slurm cluster name: $SLURM_CLUSTER_NAME"
echo "Slurm job partition: $SLURM_JOB_PARTITION"
echo "Name of the node running the job script: $SLURMD_NODENAME"

echo "Executed .py file: $1"

python -m $1

echo "End of run: $(date +%Y%m%d_%H:%M)"