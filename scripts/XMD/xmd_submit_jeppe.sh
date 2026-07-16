#!/bin/bash
#SBATCH -n 100                       # Number of cores
#SBATCH --mem=600G                    # Memory per node
#SBATCH -t 12:00:00                 # Time limit (hh:mm:ss)
#SBATCH -J jeppe_XMD_%a           # Job name                  # Memory per node
#SBATCH -o /users/lhay/scratch/jeppe_XMD_%a.out             # Standard output log
#SBATCH -e /users/lhay/scratch/jeppe_XMD_%a.err             # Standard error log
#SBATCH --partition=batch

# Load python venv
source /users/lhay/negweights/bin/activate

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

#Run python script
python3 -u xmd_calc.py --jeppe --ttbar --output="ttbar_100k_xmd_ANDERSON.npy"
