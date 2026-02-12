#!/bin/bash
#SBATCH --array=5,7,10,12,15,18,20,25,30,150
#SBATCH -J had_%a_bigR            # Job name
#SBATCH -n 104                        # Number of cores
#SBATCH --partition=batch
#SBATCH --mem=700G                    # Memory per node
#SBATCH -t 72:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/lhay/scratch/had_%agev_bigR_%j.out             # Standard output log
#SBATCH -e /users/lhay/scratch/had_%agev_bigR_%j.err             # Standard error log

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

# Load python venv
source /users/lhay/negweights/bin/activate

#Run python script
python3 -u 100k_cell_reweighting.py --max_radius=${SLURM_ARRAY_TASK_ID} --path="100k_had_emd_reweight_${SLURM_ARRAY_TASK_ID}gev_bigR.npy" --vptree='/users/lhay/cres-distance/vptree_pkls/zjet_100k_had_vptree.pkl' --points="/users/lhay/data/rjain/ppzjj_100k/hadronization_points.npy" --weights="/oscar/data/mleblan6/rjain/ppzjj_100k/weight_100k.npy" --whattype=2
