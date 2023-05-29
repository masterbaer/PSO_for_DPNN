#!/bin/bash

#SBATCH --job-name=pso_small_nn_hyperparameter
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1 # number of requested GPUs (GPU nodes shared btwn multiple jobs)
#SBATCH --ntasks=4
#SBATCH --time=45:00 # wall-clock time limit
#SBATCH --mem=128000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zx9254@partner.kit.edu


module purge                                        # Unload currently loaded modules.
module load compiler/gnu/10.2
module load devel/cuda/10.2

source venv/bin/activate      # Activate your virtual environment.

unset SLURM_NTASKS_PER_TRES

srun python -u sequential_learning/particle_swarm_opt/experiment3_hyperparameter.py