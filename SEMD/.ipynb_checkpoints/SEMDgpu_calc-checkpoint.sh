#!/bin/bash
#SBATCH -t 96:00:00 --mem=192g -p gpu --gres=gpu:2 -o "/users/lhay/scratch/unbal_sh_SEMD.out" -e "/users/lhay/scratch/unbal_sh_SEMD_err.txt" 
echo "Starting distance matrix calculation"

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh

conda activate wass_venv

python3 -u  SPECTER_calc_distance_matrix.py  --filepath "/oscar/data/mleblan6/rjain/ppzjj_100k/ppzjj_NLO_100k.root" --weight_filepath "/oscar/data/mleblan6/rjain/ppzjj_100k/weight_100k.npy"  --N 100 -sh


