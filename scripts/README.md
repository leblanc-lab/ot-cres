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

Outputs `<output_weights.npy>` plus `cell_radius_*gev.png` / `cell_pop_*gev.png` in the current directory and `neg_cell_pop_*gev.png` in `../plots/`.

### `100k_cell_reweighting_b2.py`

A beta=2 variant with the same CLI arguments as above. Differences: `compute_emds` returns `sqrt(EMD)`, event weights are rescaled by `1.455e4` (to match the 10M-event reference sample), extra progress/debug printing, and the vp-tree neighbor lookup is also saved to `vptree_links_R<radius>_b<beta>.npy` for reuse.

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

`brute_force_rw.py` performs the same `cell_reweight` procedure but from a precomputed full distance matrix instead of a vp-tree — useful when the M x N distance matrix between negative and all events is small enough to precompute directly. It loads a `.h5` distance matrix and `weight_100k.npy`, symmetrizes the matrix, scans `radii = np.logspace(1, 3, 50)`, and saves the reweighted-weight arrays and radii to disk. Paths are currently hardcoded for the "Jeppe" 100k sample — edit them for other datasets. Submit via [brute_force_rw.sh](brute_force_rw.sh).

-------------------------------------

## 4. Plotting

`plotting.py` is a module of plotting helper functions (`plot_same_rw_all`, `plot_diff_rw`, `plot_diff_samples`) that compare observable distributions before and after reweighting, saving figures to `../plots/<channel>/`. It's imported by the notebooks in [../notebooks/](../notebooks/) rather than run directly.
