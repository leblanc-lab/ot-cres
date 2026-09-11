"""Generate a small synthetic event sample for exercising the cell-reweighting workflow.

The output mimics the real inputs: a points array of shape (N_events, N_max_particles, 3)
holding [pT, eta, phi] per particle (zero-padded, leading particle first) and a weights
array of length N_events with a configurable fraction of negative weights. The kinematics
are random and carry no physics; the sample exists only so that every script in this
repository can be run end to end without access to the original samples.

    python3 make_toy_data.py --outdir ../../data/toy_example --nevents 500
"""
import argparse
import os

import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--outdir", type=str, required=True, help="Directory for hadronization_points.npy and weights.npy")
parser.add_argument("--nevents", type=int, default=500, help="Number of events (default 500)")
parser.add_argument("--max_particles", type=int, default=12, help="Maximum particles per event (default 12)")
parser.add_argument("--neg_frac", type=float, default=0.2, help="Fraction of negatively weighted events (default 0.2)")
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
N, M = args.nevents, args.max_particles

points = np.zeros((N, M, 3))
for i in range(N):
    n = rng.integers(3, M + 1)
    pt = np.sort(rng.exponential(20.0, size=n))[::-1]        # leading particle first
    eta = rng.uniform(-4.5, 4.5, size=n)
    phi = rng.uniform(-np.pi, np.pi, size=n)
    points[i, :n, 0] = pt
    points[i, :n, 1] = eta
    points[i, :n, 2] = phi

weights = rng.lognormal(mean=0.0, sigma=0.3, size=N)
weights[rng.random(N) < args.neg_frac] *= -1.0

os.makedirs(args.outdir, exist_ok=True)
np.save(os.path.join(args.outdir, "hadronization_points.npy"), points)
np.save(os.path.join(args.outdir, "weights.npy"), weights)
print(f"Wrote {N} events ({(weights < 0).sum()} negative, sum of weights = {weights.sum():.2f}) to {args.outdir}")
