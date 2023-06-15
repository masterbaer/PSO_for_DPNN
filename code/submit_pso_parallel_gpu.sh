#!/bin/bash

#SBATCH --job-name=pso_parallel_gpu
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:4 # number of requested GPUs (GPU nodes shared btwn multiple jobs)
#SBATCH --ntasks=4
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=20:00 # wall-clock time limit
#SBATCH --mem=128000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zx9254@partner.kit.edu


module purge
module load compiler/gnu/11.2
module load mpi/openmpi/4.1
module load devel/cuda/10.2

export IBV_FORK_SAFE=1
export OMPI_MCA_mpi_warn_on_fork=0
export OMPI_MCA_btl_openib_warn_default_gid_prefix=0

source venv/bin/activate      # Activate your virtual environment.

srun python -u dataparallel_learning/particle_swarm_opt/main.py