# Cell Reweighting Project

With the goal of reducing the fraction of negative weights in generated MC samples, we employ cell-reweighting [1](https://arxiv.org/abs/2109.07851) with an EMD [2](https://arxiv.org/abs/1902.02346) metric.

This repository accompanies the paper

> R. Doherty, L. Hay, R. Jain, M. LeBlanc, J. Marrinan, C. Mauceri, J. Roloff,
> *Optimal-Transport-Based Cell Resampling for Negative and Pathological Event Weights*,
> [arXiv:2607.08723](https://arxiv.org/abs/2607.08723) (2026).

If you use this code, please cite the paper; machine-readable citation metadata is in [CITATION.cff](CITATION.cff). The code is released under the [MIT License](LICENSE).

-------------------------------------

## Setup

A python virtual environment should be created from the provided requirements.txt file like so:
```
python -m venv negweights
source negweights/bin/activate
pip install -r requirements.txt
```

When returning to the virtual environment, before running any of the python scripts, activate it via:
```
source negweights/bin/activate
```

-------------------------------------

## Repository Structure

- `scripts/` — Core workflow: build a vp-tree of particle coordinates (or _points_) from a `.root` file, perform the cell reweighting, and plot the results. See [scripts/README.md](scripts/README.md) for usage details.
  - `SEMD/` — SLURM/GPU scripts that compute SPECTER-based EMD ("SEMD") distance matrices using JAX.
  - `XMD/` — Scripts and submission jobs that compute XMD, the optimal-transport distance between original and reweighted event-weight distributions, used to evaluate reweighting quality across different cell radii.
- `notebooks/` — Jupyter notebooks for tutorials and plotting, including `Intro.ipynb` (background on the cell-reweighting procedure and the EMD metric), `EMD_cartoon_zjet.ipynb` (tutorial demonstrating EMD computation for Z+jet events with EnergyFlow/POT), and `stat_power_plot.ipynb` (statistical-power scaling plot for the negative-weight fraction).
- `data/` — Inputs and intermediate results:
  - `vptree_pkls/` — Cached vp-trees produced by `make_vptree.py`. These pickles are large and regenerable, so they are not tracked in git.
  - `xmd_files/` — Precomputed XMD result arrays (`.npy`) consumed by the XMD notebooks.
  - `reweighted_files/` — Reweighted event-weight arrays (`.npy`) produced by the cell-reweighting scripts.
- `plots/` — Output plots from the reweighting and plotting scripts, organized by sample (e.g. `TTbar`, `Zjets`).

-------------------------------------

## Workflow

Cell reweighting in theory performs better the more events one has to sample from, and the radius can grow arbitrarily small. However, the computing time for cell resampling grows quadratically with the number of events, so for statistically significant samples, batched jobs are recommended.

An example batch script for reweighting lives at [scripts/reweighting/reweight_100k.sh](scripts/reweighting/reweight_100k.sh).

## Funding

This material is based on work supported by the U.S. Department of Energy, Office of Science, Office of High Energy Physics under Award Number DE-SC0026285. This work is supported by the National Science Foundation under Cooperative Agreement PHY-2019786 (The NSF AI Institute for Artificial Intelligence & Fundamental Interactions, http://iaifi.org/). We are grateful for the support this work obtained in its early stages in the form of a seed grant from the Brown University Data Science Institute.