#!/bin/bash
#SBATCH --job-name=coarse_graining
#SBATCH --output=CG_output_%j.txt      # STDOUT + STDERR output file (%j = job ID)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Number of CPU cores for Numba
#SBATCH --time=02:00:00                # Job time limit (HH:MM:SS)
#SBATCH --mem=16G                      # Memory allocation
#SBATCH --partition=standard           # Change as appropriate for your cluster

# Load modules or activate your Python environment
# Example: module load python/3.9
# Or: source ~/myenv/bin/activate
# Replace below with your actual environment setup
module load python/3.10

# Set Numba environment variable to specify CPU threads
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optional: Set thread affinity (can improve performance on some systems)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the Python script
python compute_CG.py
