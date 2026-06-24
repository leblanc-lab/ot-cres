#!/bin/bash
#SBATCH -J vptree_had_b2_zjet_100k            # Job name
#SBATCH -n 64                       # Number of cores
#SBATCH --partition=batch
#SBATCH --mem=128G                    # Memory per node
#SBATCH -t 48:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/lhay/scratch/vptree_had_b2_100k%j.out             # Standard output log
#SBATCH -e /users/lhay/scratch/vptree_had_b2_100k%j.err             # Standard error log

# Load venv
source /users/lhay/negweights/bin/activate

#Run python script
python3 -u /users/lhay/cres-distance/scripts/make_vptree.py --path="/users/lhay/cres-distance/data/vptree_pkls/zjet_100k_had_b2_vptree.pkl" --point_path="/users/lhay/data/rjain/ppzjj_100k/hadronization_points.npy" --beta=2
