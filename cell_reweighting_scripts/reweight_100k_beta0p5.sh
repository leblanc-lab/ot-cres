#!/bin/bash
#SBATCH --array=0-5
#SBATCH -J had_%a_beta0p5            # Job name
#SBATCH -n 128 # Number of cores
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --mem=400G                    # Memory per node
#SBATCH -t 72:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/lhay/scratch/had_%agev_beta0p5_%j.out             # Standard output log
#SBATCH -e /users/lhay/scratch/had_%agev_beta0p5_%j.err             # Standard error log

ARGS=(1 80 60 51.6 41.4 33.2)
ARG_STR=( 1 80 60 51p6 41p4 33p2)
echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

# Load python venv
source /users/lhay/negweights/bin/activate

#Run python script
python3 -u /users/lhay/cres-distance/cell_reweighting_scripts/100k_cell_reweighting.py --max_radius=${ARGS[SLURM_ARRAY_TASK_ID]} --path="100k_had_emd_b0p5_reweight_${ARG_STR[SLURM_ARRAY_TASK_ID]}gev.npy" --vptree='/users/lhay/cres-distance/vptree_pkls/zjet_100k_had_b0p5_vptree.pkl' --points="/oscar/data/mleblan6/rjain/ppzjj_100k/hadronization_points.npy" --weights="/oscar/data/mleblan6/rjain/ppzjj_100k/weight_100k.npy" --whattype=2 --beta=0.5