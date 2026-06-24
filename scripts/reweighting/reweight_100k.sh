#!/bin/bash
#SBATCH --array=0-2
#SBATCH -J had_%a_bigR            # Job name
#SBATCH -n 104                        # Number of cores
#SBATCH --partition=batch
#SBATCH --mem=400G                    # Memory per node
#SBATCH -t 72:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/lhay/scratch/had_%agev_bigR_%j.out             # Standard output log
#SBATCH -e /users/lhay/scratch/had_%agev_bigR_%j.err             # Standard error log

ARG_STR=(9p7 12p7 16p4)
ARGS=(9.7 12.7 16.4)

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

# Load python venv
source /users/lhay/negweights/bin/activate

#Run python script
python3 -u reweighting/100k_cell_reweighting.py --max_radius=${ARGS[SLURM_ARRAY_TASK_ID]} --path="100k_had_emd_reweight_${ARG_STR[SLURM_ARRAY_TASK_ID]}gev_bigR.npy" --vptree='/users/lhay/cres-distance/data/vptree_pkls/zjet_100k_had_vptree.pkl' --points="/users/lhay/data/rjain/ppzjj_100k/hadronization_points.npy" --weights="/oscar/data/mleblan6/rjain/ppzjj_100k/weight_100k.npy" --whattype=2 --beta=1
