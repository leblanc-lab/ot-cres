#!/bin/bash
#SBATCH --array=1,15,30,18,25,50,24,100
#SBATCH -J ps_%aGeV_ttbar_10k_bigR            # Job name
#SBATCH -n 104                       # Number of cores
#SBATCH -N 1                      #Number of nodes
#SBATCH --partition=batch
#SBATCH --mem=700G                    # Memory per node
#SBATCH -t 96:00:00                 # Time limit (hh:mm:ss)
#SBATCH --output=/users/lhay/scratch/ps_%agev_bigR_ttbar_%j.out             # Standard output log
#SBATCH --error=/users/lhay/scratch/ps_%agev_bigR_ttbar_%j.err             # Standard error log

echo "Starting job ${SLURM_ARRAY_TASK_ID} on $HOSTNAME"

# Load venv
source /users/lhay/negweights/bin/activate

#Run python script
python3 -u 100k_cell_reweighting.py --max_radius=${SLURM_ARRAY_TASK_ID} --path="/users/lhay/cres-distance/reweighted_files/10k_ps_emd_reweight_${SLURM_ARRAY_TASK_ID}gev_ttbar.npy" --vptree='/users/lhay/cres-distance/vptree_pkls/10k_ps_ttbar_vptree.pkl' --points="/users/lhay/NegativeWeights/showered_ttbar_points10k.npy" --weights="/users/lhay/NegativeWeights/ttbar_weight_10k.npy" --whattype=2 --beta=2
