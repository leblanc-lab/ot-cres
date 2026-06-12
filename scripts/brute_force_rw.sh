#!/bin/bash
#SBATCH -J jeppe_rw            # Job name
#SBATCH -n 64 # Number of cores
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --mem=360G                    # Memory per node
#SBATCH -t 24:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/lhay/scratch/jepper_rw_%j.out             # Standard output log
#SBATCH -e /users/lhay/scratch/jepper_rw_%j.err             # Standard error log

# Load python venv
source /users/lhay/negweights/bin/activate

#Run python script
python3 -u /users/lhay/cres-distance/scripts/brute_force_rw.py