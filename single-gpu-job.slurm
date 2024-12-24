#!/bin/bash
#SBATCH --job-name=single-gpu-final     # create a short name for your job
#SBATCH --partition=gpu
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:a40:1         # number of allocated gpus per node
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)

module purge

eval "$(conda shell.bash hook)"

conda activate torch-env

srun python main.py