#!/bin/bash
#SBATCH --array=5,15,18,19,20,25,30,100
#SBATCH -J hp_%aGeV_ttbar_10k_bigR            # Job name
#SBATCH -n 104                       # Number of cores
#SBATCH -N 1                      #Number of nodes
#SBATCH --partition=batch
#SBATCH --mem=700G                    # Memory per node
#SBATCH -t 96:00:00                 # Time limit (hh:mm:ss)
#SBATCH --output=/users/lhay/scratch/hp_%agev_bigR_ttbar_%j.out             # Standard output log
#SBATCH --error=/users/lhay/scratch/hp_%agev_bigR_ttbar_%j.err             # Standard error log

echo "Starting job ${SLURM_ARRAY_TASK_ID} on $HOSTNAME"

# Load venv
source /users/lhay/negweights/bin/activate

#Run python script
python3 -u 100k_cell_reweighting.py --max_radius=${SLURM_ARRAY_TASK_ID} --path="10k_hp_emd_reweight_${SLURM_ARRAY_TASK_ID}gev_ttbar.npy" --vptree='/users/lhay/cres-distance/vptree_pkls/10k_hp_ttbar_vptree.pkl' --points="/users/lhay/NegativeWeights/hardprocess_ttbar_points10k.npy" --weights="/users/lhay/NegativeWeights/ttbar_weight_100k.npy" --whattype=0
