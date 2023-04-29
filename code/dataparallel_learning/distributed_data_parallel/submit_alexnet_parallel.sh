#!/bin/bash

#SBATCH --job-name=alex4
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:4 # number of requested GPUs (GPU nodes shared btwn multiple jobs)
#SBATCH --ntasks=4
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=45:00 # wall-clock time limit
#SBATCH --mem=128000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zx9254@partner.kit.edu


module purge                                        # Unload currently loaded modules.
module load compiler/gnu/10.2
module load devel/cuda/10.2

source <path to your venv folder>/bin/activate      # Activate your virtual environment. TODO

unset SLURM_NTASKS_PER_TRES

srun python -u <path to your python script>/main.py #TODO