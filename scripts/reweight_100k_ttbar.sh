#!/bin/bash
#SBATCH --array=0,1
#SBATCH -J ps_%aGeV_ttbar_100k            # Job name
#SBATCH -n 104                       # Number of cores
#SBATCH -N 1                      #Number of nodes
#SBATCH --partition=batch
#SBATCH --mem=700G                    # Memory per node
#SBATCH -t 96:00:00                 # Time limit (hh:mm:ss)
#SBATCH --output=/users/lhay/scratch/ps_%agev_bigR_ttbar_%j.out             # Standard output log
#SBATCH --error=/users/lhay/scratch/ps_%agev_bigR_ttbar_%j.err             # Standard error log

echo "Starting job ${SLURM_ARRAY_TASK_ID} on $HOSTNAME"

ARG_STR=(37 38)
ARGS=(37 38)

# Load venv
source /users/lhay/negweights/bin/activate

#Run python script
python3 -u 100k_cell_reweighting.py --max_radius=${ARG[SLURM_ARRAY_TASK_ID]} --path="100k_ps_emd_reweight_${ARG_STR[SLURM_ARRAY_TASK_ID]}gev_ttbar.npy" --vptree='/users/lhay/cres-distance/data/vptree_pkls/100k_ps_ttbar_vptree.pkl' --points="/oscar/data/mleblan6/lhay/ttbar_100k/showered_ttbar_points.npy" --weights="/oscar/data/mleblan6/lhay/ttbar_100k/ttbar_weight_100k.npy" --whattype=1
