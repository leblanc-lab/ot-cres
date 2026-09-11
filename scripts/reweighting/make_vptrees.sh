#!/bin/bash
#CHANGE THESE TO MATCH YOUR SPECIFIC JOB
#SBATCH -J vptree_had_ttbar_100k            # Job name
#SBATCH -n 64                        # Number of cores
#SBATCH --partition=batch
#SBATCH --mem=64G                    # Memory per node
#SBATCH -t 96:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/lhay/scratch/vptree_had_ttbar_100k_%j.out             # Standard output log
#SBATCH -e /users/lhay/scratch/vptree_had_ttbar_100k_%j.err             # Standard error log

# Load Conda
source /users/lhay/negweights/bin/activate

#Run python script
python3 -u make_vptree.py --path='100k_had_ttbar_vptree.pkl' --point_path="/users/lhay/NegativeWeights/hadronization_ttbar_points.npy"
