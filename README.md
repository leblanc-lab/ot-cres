# cres-distance
# Cell Reweighting Project

With the goal of reducing the fraction of negative weights in generated MC samples, we employ cell-reweighting [1](https://arxiv.org/abs/2109.07851) with an EMD [2](https://arxiv.org/abs/1902.02346) metric.

-------------------------------------

Scripts to produce a map of nearest neighbors, or vptree, from particle coordinates (or _points_), starting from a `.root` file, and then perform the reweighitng, can be can be found in `/cell_reweighting_scripts`.

------------------------------------

A python virtual environment should be created from the provided requirements.txt file like so:
```
python -m venv negweights
pip install -r requirements.txt
```

Then scripts can be ran.
When returning to the virtual environment, before running any of the python scripts, one should activate the environment via:

```
source negweights/bin/activate
```

------------------------------------

Cell reweighting in theory performs better the more events one has to sample from, and the radius can grow arbitrarily small. However, the computing time for cell resampling grows quadratically with time, so for statistically significent samples, batched jobs are recommended.

An example of batch script for reweighting lives [here](https://github.com/laurenhay/cres-distance/blob/main/cell_reweighting_scripts/reweight_100k.sh).

