# Cell Reweighting Project

With the goal of reducing the fraction of negative weights in generated MC samples, we employ cell-reweighting [1](https://arxiv.org/abs/2109.07851) with an EMD [2](https://arxiv.org/abs/1902.02346) metric.

-------------------------------------

## Setup

A python virtual environment should be created from the provided requirements.txt file like so:
```
python -m venv negweights
pip install -r requirements.txt
```

When returning to the virtual environment, before running any of the python scripts, activate it via:
```
source negweights/bin/activate
```

-------------------------------------

## Repository Structure

- `cell_reweighting_scripts/` — Core workflow: build a vp-tree of particle coordinates (or _points_) from a `.root` file, perform the cell reweighting, and plot the results. See [cell_reweighting_scripts/README.md](cell_reweighting_scripts/README.md) for usage details.
- `SEMD/` — SLURM/GPU scripts that compute SPECTER-based EMD ("SEMD") distance matrices using JAX.
- `XMD_scripts/` — Scripts and notebooks that compute XMD, the optimal-transport distance between original and reweighted event-weight distributions, used to evaluate reweighting quality across different cell radii.
- `xmd_files/` — Precomputed XMD result arrays (`.npy`) consumed by the XMD notebooks.
- `vptree_pkls/` — Cached vp-trees produced by `make_vptree.py`. These pickles are large and regenerable, so they are not tracked in git.
- `plots/` — Output plots from the reweighting and plotting scripts, organized by sample (e.g. `TTbar`, `Zjets`).
- `plot_observable.py`, `plot_cell_radius_hist.py` — Top-level scripts that plot kinematic observables and cell-radius histograms from distance matrices and event-weight files.
- `make_vptrees.sh` — Example SLURM batch script for building a vp-tree.
- `Intro.ipynb` — Background on the cell-reweighting procedure and the EMD metric.
- `EMD_cartoon_zjet.ipynb` — Tutorial demonstrating EMD computation for Z+jet events with EnergyFlow/POT.
- `stat_power_plot.ipynb` — Statistical-power scaling plot for the negative-weight fraction (`negative_fraction_scaling.png`).

-------------------------------------

## Workflow

Cell reweighting in theory performs better the more events one has to sample from, and the radius can grow arbitrarily small. However, the computing time for cell resampling grows quadratically with the number of events, so for statistically significant samples, batched jobs are recommended.

An example batch script for reweighting lives [here](https://github.com/laurenhay/cres-distance/blob/main/cell_reweighting_scripts/reweight_100k.sh).
