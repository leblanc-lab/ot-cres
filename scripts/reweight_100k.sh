#!/bin/bash
#SBATCH --array=0-6
#SBATCH -J ps_%a_bigR            # Job name
#SBATCH -n 104                        # Number of cores
#SBATCH --partition=batch
#SBATCH --mem=400G                    # Memory per node
#SBATCH -t 72:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/lhay/scratch/ps_%agev_bigR_%j.out             # Standard output log
#SBATCH -e /users/lhay/scratch/ps_%agev_bigR_%j.err             # Standard error log

ARG_STR=(1 40 17 12 9 12p4 16p7)
ARGS=(1 40 17 12 9 12.4 16.7)

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

# Load python venv
source /users/lhay/negweights/bin/activate

#Run python script
python3 -u 100k_cell_reweighting.py --max_radius=${ARGS[SLURM_ARRAY_TASK_ID]} --path="100k_ps_emd_reweight_${ARG_STR[SLURM_ARRAY_TASK_ID]}gev_bigR.npy" --vptree='/users/lhay/cres-distance/vptree_pkls/zjet_100k_ps_vptree.pkl' --points="/users/lhay/data/rjain/ppzjj_100k/showered_points.npy" --weights="/oscar/data/mleblan6/rjain/ppzjj_100k/weight_100k.npy" --whattype=2 --beta=1
