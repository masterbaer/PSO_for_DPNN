#!/bin/bash

#SBATCH --job-name=mpi_test        # job name
#SBATCH --partition=multiple            # queue for resource allocation
#SBATCH --nodes=4                       # number of nodes to be used
#SBATCH --time=30:00                    # wall-clock time limit
#SBATCH --mem=4000                     # memory per node
#SBATCH --cpus-per-task=1              # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1             # maximum count of tasks per node
#SBATCH --mail-type=ALL                 # Notify user by email when certain event types occur.
#SBATCH --mail-user=zx9254@partner.kit.edu

# Set up modules.
module purge
module load compiler/gnu/11.2
module load mpi/openmpi/4.1
module load devel/cuda/10.2

source venv/bin/activate      # Activate your virtual environment.

mpirun python -u dataparallel_learning/particle_swarm_opt/experiment9_overestimation.py
