#import useful stuff                                                                                                                                                                                                             
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
#check number of cores     

num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Cell reweighting using distance matrices")

    # ---------- Inputs ----------
    parser.add_argument("--matrix",type=str, required=True,help="Path to HDF5 or .npy distance matrix",)

    parser.add_argument("--weights",type=str,required=True,help="Path to numpy weights file (.npy)",)

    parser.add_argument("--dataset",type=str,default="distance_matrix",help="Dataset name inside HDF5 file",)

    # ---------- Output ----------
    parser.add_argument("--output",type=str,required=True,help="Output npy filename",)

    # ---------- Radius options ----------
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument( "--radius",type=float,help="Single reweighting radius",)

    group.add_argument("--radii",type=float,nargs="+",help="Explicit list of radii",)

    group.add_argument("--logspace",type=float,nargs=3,metavar=("MIN", "MAX", "N"),help="Generate radii using np.logspace(log10(MIN), log10(MAX), N)",)

    # ---------- Reweight settings ----------
    parser.add_argument("--weight-tol",type=float,default=0.01,help="Cell weight stopping tolerance",)

    # ---------- Debug/testing ----------
    parser.add_argument("--max-events",type=int,default=None,help="Only use first N events (testing)",)

    parser.add_argument("--verbose",action="store_true",help="Enable verbose printing",)

    return parser.parse_args()


def sort_dist(orig_dist_matrix, neg_events, N):
    sorted_dist_matrix = np.zeros((len(neg_events), N))
    for i in range(len(neg_events)):
        sorted_dist_matrix[i] = np.argsort(orig_dist_matrix[i])
        if i % 2000 == 0:
            print(i)
    
    return sorted_dist_matrix

def cell_reweight(max_radius, event_weights, dist_matrix, verbose = False, weight_tol = 0.01):
    """dist_matrix must have shape (N_negative_events, N_events): row i is the distance from neg_events[i] to every event."""
    neg_events = np.where(event_weights < 0)[0]
    N = len(event_weights)
    reweighted_event_weight = np.copy(np.array(event_weights))
    cell_radius = []
    sorted_dist_matrix = sort_dist(dist_matrix, neg_events, N)
    if sorted_dist_matrix.shape[1] != N:
        raise ValueError(
            f"dist_matrix has {sorted_dist_matrix.shape[1]} columns, "
            f"but event_weights has length {N}"
        )

    if sorted_dist_matrix.shape[0] != len(neg_events):
        raise ValueError(
            f"dist_matrix has {sorted_dist_matrix.shape[0]} rows, "
            f"but neg_events has length {len(neg_events)}. "
            "This function assumes row i corresponds to neg_events[i]."
        )
        
    for neg_event_idx, event_idx in enumerate(neg_events):
        cell_weight = reweighted_event_weight[event_idx]
        abs_cell_weight = abs(reweighted_event_weight[event_idx])
        events_in_cell = [event_idx]
        final_cell_event = None

        if reweighted_event_weight[event_idx] >= 0:
            #### this means the event was already rw'd in a different cell
            continue
        
        for j in range(N):
            close2neg = int(sorted_dist_matrix[neg_event_idx, j])
            if close2neg == event_idx:
                continue
            distance = dist_matrix[neg_event_idx, close2neg]
            if (cell_weight <= weight_tol) and (distance < max_radius):
                neighbor_weight = reweighted_event_weight[close2neg]
                cell_weight += neighbor_weight
                abs_cell_weight += abs(neighbor_weight)
                events_in_cell.append(close2neg)
                final_cell_event = close2neg

            else:
                break


        if cell_weight > 0 and final_cell_event is not None:
            cell_radius = np.append(
                cell_radius,
                dist_matrix[neg_event_idx, final_cell_event]
            )
            reweighted_event_weight[events_in_cell] = (
                cell_weight / abs_cell_weight *
                abs(reweighted_event_weight[events_in_cell])
            )

    
    #print('Hard process reweighting completed')
    frac_rw = 1- (len(np.where(reweighted_event_weight < 0)[0]) / len(neg_events))
    print(f'{frac_rw*100:.1f}% of negative weights are reweighted at {max_radius} GeV cell radius')

    return reweighted_event_weight




def load_dist_matrix(path, dataset="distance_matrix", max_events=None):
    """Load a (possibly upper-triangular) N x N distance matrix from .h5 or .npy and symmetrize it."""
    if path.endswith(".h5") or path.endswith(".hdf5"):
        with h5py.File(path, "r") as f:
            m = f[dataset][:max_events, :max_events] if max_events else f[dataset][:]
    else:
        m = np.load(path)
        if max_events:
            m = m[:max_events, :max_events]
    m = np.array(m, dtype=float)
    if m.shape[0] != m.shape[1]:
        raise ValueError(f"expected a square N x N matrix, got {m.shape}")
    # fill the lower triangle from the upper one (no-op if already symmetric)
    iu = np.triu_indices_from(m, k=1)
    m[iu[1], iu[0]] = m[iu]
    return m


def main():
    args = parse_args()

    print("Loading event weights")
    event_weights = np.load(args.weights)
    if args.max_events is not None:
        event_weights = event_weights[:args.max_events]
    N = len(event_weights)

    print("Loading distance matrix")
    full_dist_matrix = load_dist_matrix(args.matrix, args.dataset, args.max_events)
    if full_dist_matrix.shape[0] < N:
        raise ValueError(f"distance matrix has {full_dist_matrix.shape[0]} rows but there are {N} weights")
    full_dist_matrix = full_dist_matrix[:N, :N]
    # cell_reweight expects one row per negative event
    neg_events = np.where(event_weights < 0)[0]
    dist2neg = full_dist_matrix[neg_events]

    if args.radius is not None:
        radii = np.array([args.radius])
    elif args.radii is not None:
        radii = np.array(args.radii)
    else:
        rmin, rmax, n = args.logspace
        radii = np.logspace(np.log10(rmin), np.log10(rmax), int(n))

    reweights = np.empty((0, N))
    for r in radii:
        reweights = np.vstack((reweights, cell_reweight(r, event_weights, dist2neg, verbose=args.verbose, weight_tol=args.weight_tol)))

    out = args.output if args.output.endswith(".npy") else args.output + ".npy"
    np.save(out, reweights)
    np.savetxt(out[:-4] + "_radii.txt", radii)
    print("Saved reweights to", out, "and radii to", out[:-4] + "_radii.txt")


if __name__ == "__main__":
    main()
