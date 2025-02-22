#!/bin/bash
#SBATCH --job-name=ddp-testing   # create a short name for your job
#SBATCH --partition=gpu
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:a40:2         # number of allocated gpus per node
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=%j.out
#SBATCH --error=%j.err           # Separate error file

set -x
nvidia-smi

export MASTER_PORT='12355'
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"

conda activate final-env

srun python -u main.py -ns 1000000 -e 40

pkill -9 python