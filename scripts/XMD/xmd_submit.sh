#!/bin/bash
#SBATCH -n 64                        # Number of cores
#SBATCH --mem=700G                    # Memory per node
#SBATCH -t 36:00:00                 # Time limit (hh:mm:ss)
#SBATCH --array=1,2
#SBATCH -J ttbar_XMD_%a           # Job name                  # Memory per node
#SBATCH -o /users/lhay/scratch/ttbar_XMD_%a.out             # Standard output log
#SBATCH -e /users/lhay/scratch/ttbar_XMD_%a.err             # Standard error log
#SBATCH --partition=batch

ARGS=("hp" "ps" "had")
# Load python venv
source /users/lhay/negweights/bin/activate

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

#Run python script
python3 -u xmd_calc.py --ttbar --stage=$SLURM_ARRAY_TASK_ID --output="ttbar_100k_xmd_${ARGS[SLURM_ARRAY_TASK_ID]}.npy"
