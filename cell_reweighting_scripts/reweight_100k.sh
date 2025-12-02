#CHANGE THESE TO MATCH YOUR SPECIFIC JOB
#!/bin/bash
#SBATCH -J had_17p2_bigR            # Job name
#SBATCH -n 64                        # Number of cores
#SBATCH --partition=batch
#SBATCH --mem=700G                    # Memory per node
#SBATCH -t 96:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/rrjain/scratch/had_17p2gev_bigR_%j.out             # Standard output log
#SBATCH -e /users/rrjain/scratch/had_17p2gev_bigR_%j.err             # Standard error log

# Load Conda
source /oscar/runtime/software/external/miniforge/23.11.0-0/etc/profile.d/conda.sh

# Activate venv CHANGE THIS TO MATCH YOUR VENV NAME
conda activate wass

#Run python script
python3 -u 100k_cell_reweighting.py --max_radius=17.2 --path='100k_had_emd_reweight_17p2gev_bigR.npy' --vptree='vptree_had_100k_bigR.pkl' --whattype=2
