#!/bin/bash
# End-to-end smoke test of the cell-reweighting workflow on a synthetic sample.
# Run from anywhere; writes everything under <repo>/data/toy_example and <repo>/data/vptree_pkls.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA="$REPO/data/toy_example"
NEVENTS="${NEVENTS:-500}"
NPROC="${NPROC:-4}"

echo "== 1. toy sample"
python3 "$HERE/make_toy_data.py" --outdir "$DATA" --nevents "$NEVENTS"

echo "== 2. vp-tree (beta=1)"
mkdir -p "$REPO/data/vptree_pkls"
python3 "$REPO/scripts/reweighting/make_vptree.py" \
    --path="$REPO/data/vptree_pkls/toy_had_vptree.pkl" \
    --point_path="$DATA/hadronization_points.npy" --beta=1

echo "== 3. cell reweighting with the vp-tree at two radii"
for R in 20 50; do
    python3 "$REPO/scripts/reweighting/100k_cell_reweighting.py" \
        --vptree="$REPO/data/vptree_pkls/toy_had_vptree.pkl" \
        --points="$DATA/hadronization_points.npy" --weights="$DATA/weights.npy" \
        --path="$DATA/toy_had_emd_reweight_${R}gev.npy" \
        --max_radius="$R" --whattype=2 --beta=1 --nproc="$NPROC"
done

echo "== 4. full EMD distance matrix (for brute-force reweighting and XMD)"
python3 "$REPO/scripts/XMD/compute_distmatrix.py" \
    --points="$DATA/hadronization_points.npy" --output="$DATA/toy_distmatrix.npy"

echo "== 5. brute-force reweighting from the distance matrix at the same radii"
python3 "$REPO/scripts/reweighting/brute_force_rw.py" \
    --matrix="$DATA/toy_distmatrix.npy" --weights="$DATA/weights.npy" \
    --radii 20 50 --output="$DATA/toy_bruteforce_reweights.npy"

echo "== 6. XMD between original and reweighted samples"
python3 "$REPO/scripts/XMD/xmd_calc.py" \
    --weights_orig="$DATA/weights.npy" --distmatrix="$DATA/toy_distmatrix.npy" \
    --radii 20 50 --reweights="$DATA/toy_bruteforce_reweights.npy" \
    --output="$DATA/toy_xmd.npy"

echo "== done. Outputs in $DATA"
