"""Compute the full EMD (beta=1) distance matrix between a block of events and every event in the sample.

    python3 compute_distmatrix.py --points=<points.npy> --output=<out.npy> [--index=K --chunk=10000] [--nproc=N]

Without --index the whole N x N matrix is computed. With --index=K, rows K*chunk : (K+1)*chunk are computed
(one SLURM array task per block); concatenate the blocks afterwards.
"""
import numpy as np
import os
import wasserstein
import multiprocessing as mp
import argparse
import time

#parameters of EMD calculation
max_dist = np.sqrt(9.8**2 + (2*np.pi)**2)

calc_emds = wasserstein.EMDYPhi(R=max_dist, 
                                beta=1,
                                norm=False,
                                n_iter_max=100000,
                                dtype='float32')

def compute_emds2(points1, points2):
    pts1 = points1[:, 0].astype('float32')
    pts2 = points2[:, 0].astype('float32')

    etaphi1 = points1[:, 1:3].astype('float32')
    etaphi2 = points2[:, 1:3].astype('float32')

    if np.all(pts1 == 0):
        pts1[0] += 1e-5

    if np.all(pts2 == 0):
        pts2[0] += 1e-5

    return calc_emds(pts1, etaphi1, pts2, etaphi2)


# Worker-side copy of the points. Inherited from the parent under 'fork'; loaded once per worker under 'spawn'.
all_points = None

def init_worker(point_path):
    global all_points
    if all_points is None:
        all_points = np.load(point_path)

def compute_row(i):
    """Distances from event i to every event in the sample."""
    if i % 100 == 0:
        print(i, flush=True)
    row = np.empty(len(all_points), dtype='float32')
    for j in range(len(all_points)):
        try:
            row[j] = compute_emds2(all_points[i], all_points[j])
        except RuntimeError as e:
            print(f"Error with EMD calculation between event {i} and {j}: {str(e)}")
            row[j] = 1000
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type = int, default = None, required = False, help = "Which block of rows of the distance matrix to compute (default: all rows)")
    parser.add_argument("--chunk", type = int, default = 10000, help = "Number of rows per block when --index is given (default 10000)")
    parser.add_argument("--points", type = str, required = True, help = "Path of points file to run over")
    parser.add_argument("--output", type = str, required = True, help = "Path of output file (.npy)")
    parser.add_argument("--nproc", type = int, default = None, help = "Number of worker processes (default: CPUs available to this job)")
    args = parser.parse_args()

    global all_points
    all_points = np.load(args.points)
    n_all = len(all_points)

    if args.index is None:
        print("Running over all points in sample")
        start, stop = 0, n_all
    else:
        start = args.index * args.chunk
        stop = min(start + args.chunk, n_all)
        print(f"Running over points {start}:{stop}")

    if args.nproc is not None:
        nproc = args.nproc
    else:
        try:
            nproc = len(os.sched_getaffinity(0))
        except AttributeError:
            nproc = os.cpu_count()
    print(f"Number of CPU cores: {os.cpu_count()}; using {nproc} worker processes")

    start_pc = time.perf_counter()
    start_pt = time.process_time()

    with mp.Pool(processes=nproc, initializer=init_worker, initargs=(args.points,)) as pool:
        rows = pool.map(compute_row, range(start, stop), chunksize=1)

    pc = time.perf_counter()-start_pc
    pt = time.process_time() - start_pt
    print("time taken to make dist matrix ", pc ," (perf count) ", pt ," (proc time)") 

    full_distmatrix = np.array(rows, dtype='float32').reshape(stop - start, n_all)
    np.save(args.output, full_distmatrix)
    print("Saved", full_distmatrix.shape, "distance matrix to", args.output)


if __name__ == "__main__":
    main()
