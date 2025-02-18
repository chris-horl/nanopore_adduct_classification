#!/bin/bash
#SBATCH --nice=0
#SBATCH --time=03:00:00
#SBATCH --gres=gpu
#SBATCH --mem=40G
#SBATCH --partition=gpushort
#SBATCH --output="./output_training_sbatch/%j_training.out"

echo "conda env: $CONDA_PROMPT_MODIFIER"
echo "Start of run: $(date +%Y%m%d_%H:%M)"
echo "Slurm cluster name: $SLURM_CLUSTER_NAME"
echo "Slurm job partition: $SLURM_JOB_PARTITION"
echo "Name of the node running the job script: $SLURMD_NODENAME"

echo ".py file for training: $1"
echo ".yaml config file: $2"

python $1 fit --config $2

echo "End of run: $(date +%Y%m%d_%H:%M)"