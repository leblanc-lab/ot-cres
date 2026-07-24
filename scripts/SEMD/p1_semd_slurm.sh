#!/bin/bash
#SBATCH -J p1_showered          # Job name
#SBATCH -n 64                        # Number of cores
#SBATCH --partition=batch
#SBATCH --mem=492G                    # Memory per node
#SBATCH -t 2:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/rrjain/scratch/p1_showered_%j.out             # Standard output log
#SBATCH -e /users/rrjain/scratch/p1_showered_%j.err             # Standard error log

# Load Conda
module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh

# Activate venv
conda activate wass

#Run python script
python3 -u p1_semd_distmatrix.py
