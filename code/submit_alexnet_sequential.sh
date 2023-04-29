#!/bin/bash

#SBATCH --job-name=alex_cpu_sequential
#SBATCH --partition=single
#SBATCH --time=30:00 # wall-clock time limit, adapt if necessary
#SBATCH --mem=128000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zx9254@partner.kit.edu

module purge                                        # Unload currently loaded modules.
module load compiler/gnu/10.2
module load devel/cuda/10.2

source venv/bin/activate      # Activate your virtual environment.

python -u sequential_learning/gradient_descent/main.py