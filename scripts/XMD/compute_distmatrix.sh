#!/bin/bash
#SBATCH --array=0
#SBATCH --job-name=distmatrix_jobarray_%a           # Job name
#SBATCH --ntasks=360                               # Number of tasks (64 cores per job)
#SBATCH -N 2
#SBATCH --cpus-per-task=1                           # Number of CPU cores per task
#SBATCH --partition=batch
#SBATCH --mem=750G                    # Memory per node
#SBATCH -t 96:00:00                 # Time limit (hh:mm:ss)
#SBATCH --output=/users/lhay/scratch/had_trial_%j_%a.out
#SBATCH --error=/users/lhay/scratch/had_trial_%j_%a.err

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"
# Load python venv
source /users/lhay/negweights/bin/activate

#Run python script
python3 -u compute_distmatrix.py --points="/oscar/data/mleblan6/lhay/ttbar_100k/hadronization_ttbar_points.npy" --output="trial${SLURM_ARRAY_TASK_ID}_distmatrix_ttbar.npy" --index=${SLURM_ARRAY_TASK_ID}