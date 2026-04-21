#!/bin/bash
#SBATCH --array=0-3
#SBATCH -J ps_%a_betRj            # Job name
#SBATCH -n 96 # Number of cores
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --mem=700G                    # Memory per node
#SBATCH -t 72:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/lhay/scratch/ps_%agev_beta2Rj_%j.out             # Standard output log
#SBATCH -e /users/lhay/scratch/ps_%agev_beta2Rj_%j.err             # Standard error log
ARG_END=(0 1 2 3 4 5)
ARGS=(1.0 1.1 1.2 1.3 1.4 1.5)
echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

# Load python venv
source /users/lhay/negweights/bin/activate

#Run python script
python3 -u /users/lhay/cres-distance/cell_reweighting_scripts/100k_cell_reweighting.py --max_radius=${ARGS[SLURM_ARRAY_TASK_ID]} --path="100k_ps_emd_b2RJ_reweight_1p${ARG_END[SLURM_ARRAY_TASK_ID]}gev.npy" --vptree='/oscar/data/mleblan6/rjain/beta2/showered_beta2_vptree.pkl' --points="/oscar/data/mleblan6/rjain/ppzjj_100k/showered_points.npy" --weights="/oscar/data/mleblan6/rjain/ppzjj_100k/weight_100k.npy" --whattype=2 --beta=2