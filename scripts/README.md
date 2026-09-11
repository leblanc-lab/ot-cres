# Scripts

Scripts for building vp-trees, performing cell reweighting (with or without a vp-tree), and plotting the resulting observables. `SEMD/` and `XMD/` contain separate sub-workflows for computing SEMD and XMD distances.

Most `.sh` files here are SLURM batch scripts with hardcoded, user- and machine-specific paths (job names, `#SBATCH -o`/`-e` log paths, input `.npy`/`.pkl` paths, output filenames). Treat them as examples — edit the SBATCH directives and arguments before submitting with `sbatch <script>.sh`.

-------------------------------------

## 1. Building a vp-tree

`make_vptree.py` builds a vantage-point (vp) tree over a set of particle "points" (an `.npy` array of `[pT, eta, phi]` per particle, per event), using the EMD as the distance metric, and pickles it to disk.

```
python3 make_vptree.py --path=<output.pkl> --point_path=<points.npy> [--beta=1]
```

- `--path` — output path for the pickled vp-tree (must end in `.pkl`)
- `--point_path` — input `.npy` file of points (hard-process, showered, or hadronization)
- `--beta` — EMD beta parameter (default `1`)

Example SLURM submission: [vptree_job.sh](vptree_job.sh) builds a beta=2 vp-tree for the hadronization sample. [make_vptrees.sh](make_vptrees.sh) is a bare template for the same job — update the job name, resources, log paths, and `make_vptree.py` arguments before use.

-------------------------------------

## 2. Cell reweighting with a vp-tree

Given a vp-tree, event points, and event weights, these scripts find every event within `--max_radius` (in the EMD metric) of each negatively-weighted event, redistribute weight within that "cell", and save the new weights plus diagnostic plots (`cell_radius_*`, `cell_pop_*`, `neg_cell_pop_*`).

### `100k_cell_reweighting.py`

The general-purpose version:

```
python3 100k_cell_reweighting.py \
  --vptree=<vptree.pkl> --points=<points.npy> --weights=<weights.npy> \
  --path=<output_weights.npy> --max_radius=<R> --whattype=<0|1|2> [--beta=1]
```

- `--whattype` — `0` = hard process, `1` = showered, `2` = hadronization. For `whattype=0` this also drops all-zero padding rows from `points`/`weights`. Otherwise controls the title/filename convention used for the diagnostic plots.
- `--beta` — EMD beta; `beta <= 1` uses the EMD directly, `beta > 1` uses the energy-difference-corrected form
- `--nproc` — number of worker processes for the vp-tree queries (default: the CPUs available to the job)

The number of events is taken from the `points` file; the `weights` file must have at least as many entries (extra trailing entries are ignored with a warning).

Outputs `<output_weights.npy>` plus per-cell diagnostics `cell_radius_*.npy`, `cell_pop_*.npy` and `neg_cell_pop_*.npy` in `../data/cell_info/` (created if missing), named by stage, radius and beta (e.g. `neg_cell_pop_had_16p4gev_b1.npy`).

### `100k_beta0p5_reweighting.py`

A beta=0.5 variant with hardcoded input paths (`/users/jsmarrin/100k_sample/{hardprocess,showered,hadronization}_points.npy` and `weight_100k.npy`, also rescaled by `1.455e4`), selected via `--whattype`:

```
python3 100k_beta0p5_reweighting.py --vptree=<vptree.pkl> --path=<output_weights.npy> --max_radius=<R> --whattype=<0|1|2>
```

### SLURM array jobs

These submit `100k_cell_reweighting.py` as a SLURM array job, scanning a list of radii (one per array index) for a specific sample and vp-tree:

- `reweight_100k.sh` — 100k Z+jet showered sample, `--whattype=2 --beta=1`
- `reweight_100k_beta0p5.sh` — 100k Z+jet hadronization sample, `--whattype=2 --beta=0.5`
- `reweight_100k_newbeta.sh` — 100k Z+jet showered sample, `--whattype=0 --beta=2`
- `reweight_100k_ttbar.sh` — 100k ttbar showered sample, `--whattype=1`
- `reweight_10k_ttbar.sh` — 10k ttbar showered sample, `--whattype=2 --beta=2`; here the radii come directly from the SLURM array values

-------------------------------------

## 3. Brute-force reweighting (no vp-tree)

`brute_force_rw.py` performs the same `cell_reweight` procedure but from a precomputed full N x N distance matrix instead of a vp-tree — useful when the matrix is small enough to precompute directly (see `XMD/compute_distmatrix.py`). It accepts an `.h5` (upper-triangular is fine, it is symmetrized) or `.npy` matrix:

```
python3 brute_force_rw.py --matrix=<distmatrix.h5|.npy> --weights=<weights.npy> --output=<reweights.npy> \
    (--radius R | --radii R1 R2 ... | --logspace MIN MAX N) [--dataset distance_matrix] [--weight-tol 0.01] [--max-events N]
```

It saves a stacked `(N_radii, N_events)` array of reweighted weights plus `<output>_radii.txt`. `--weight-tol` is the cell-sum threshold at which the cell stops growing (default `0.01`; the vp-tree script uses `1e-5`, so set it to `1e-5` to reproduce that script exactly). Submit via [brute_force_rw.sh](brute_force_rw.sh).

-------------------------------------

## 4. Distance matrices and XMD

`XMD/compute_distmatrix.py` computes the full EMD (beta=1) matrix between a block of events and the whole sample, one row per event, in parallel:

```
python3 XMD/compute_distmatrix.py --points=<points.npy> --output=<out.npy> [--index=K --chunk=10000] [--nproc=N]
```

`XMD/xmd_calc.py` then computes the Cross-Section Mover's Distance between the original and each reweighted sample:

```
python3 XMD/xmd_calc.py --weights_orig=<weights.npy> --distmatrix=<distmatrix.npy> --radii R1 R2 ... \
    (--reweights <stacked.npy> | --reweight_prefix <prefix> [--reweight_suffix gev_bigR.npy]) --output=<xmd.npy>
```

The `--ttbar`, `--test` and `--jeppe` flags select presets with paths hardcoded for the Oscar cluster.

-------------------------------------

## 5. End-to-end example

`examples/run_toy_example.sh` generates a small synthetic sample (`examples/make_toy_data.py`), builds a vp-tree, reweights at two radii, computes the full distance matrix, repeats the reweighting by brute force, and computes the XMD. It runs in a few seconds on a laptop and exercises every script above:

```
bash scripts/examples/run_toy_example.sh        # NEVENTS=500 NPROC=4 by default
```

Outputs go to `data/toy_example/`, `data/vptree_pkls/` and `data/cell_info/` (all git-ignored).

-------------------------------------

## 6. Plotting

`plotting.py` is a module of plotting helper functions (`plot_same_rw_all`, `plot_diff_rw`, `plot_diff_samples`) that compare observable distributions before and after reweighting, saving figures to `../plots/<channel>/`. It's imported by the notebooks in [../notebooks/](../notebooks/) rather than run directly.
