import pickle 
import numpy as np
import wasserstein
import os
import argparse
import multiprocessing as mp
import warnings
import time

# ---------------------------------------------------------------------------------------------------------------------------------
# EMD distance function.
#
# It must live at module level under the name `compute_emds`: the pickled vp-tree produced by make_vptree.py
# references its distance function as `__main__.compute_emds`, and every worker process unpickles the tree.
# `configure_emd(beta)` sets the module globals it depends on and is called in the parent and in each worker.
# ---------------------------------------------------------------------------------------------------------------------------------
max_dist = np.sqrt(9.8**2 + (2*np.pi)**2)
beta = 1.0
calc_emds = None

def configure_emd(beta_value):
    global beta, calc_emds
    beta = float(beta_value)
    calc_emds = wasserstein.EMDYPhi(R=max_dist, 
                                    beta=beta,
                                    norm=False,
                                    n_iter_max=100000,
                                    dtype='float64')

def compute_emds(points1, points2):
    pts1 = points1[:, 0]
    pts2 = points2[:, 0]

    etaphi1 = points1[:, 1:3]
    etaphi2 = points2[:, 1:3]

    if beta > 1.0:
        dE = np.abs(pts1.sum() - pts2.sum())

    if np.all(pts1 == 0):
        pts1[0] += 1e-5

    if np.all(pts2 == 0):
        pts2[0] += 1e-5

    if beta <= 1.0:
        try: 
            return calc_emds(pts1, etaphi1, pts2, etaphi2)
        except RuntimeError: #sometimes the EMD optimization fails to converge 
            pts1[-1] += 1e-5
            pts2[-1] += 1e-5
            return calc_emds(pts1, etaphi1, pts2, etaphi2)
    else:
        # energy-difference-corrected form for beta > 1
        try:
            return (calc_emds(pts1, etaphi1, pts2, etaphi2) - dE)**(1.0/beta) + dE
        except Exception:
            etaphi1[0,0] += 1e-5
            etaphi2[0,0] += 1e-5
            return (calc_emds(pts1, etaphi1, pts2, etaphi2) - dE)**(1.0/beta) + dE


def load_inputs(point_filepath, weight_filepath, whattype):
    points = np.load(point_filepath)
    event_weight = np.load(weight_filepath)

    if len(event_weight) < len(points):
        raise ValueError(f"weights file has {len(event_weight)} entries but points file has {len(points)} events")
    if len(event_weight) > len(points):
        warnings.warn(f"weights file has {len(event_weight)} entries but points file has {len(points)} events; "
                      f"using only the first {len(points)} weights", UserWarning)
        event_weight = event_weight[:len(points)]

    if whattype == 0:
        # drop all-zero padding rows (events with no hard-process particles surviving the cuts)
        mask = np.where(np.all(points == 0, axis=(1, 2)))[0]
        points = np.delete(points, mask, axis = 0)
        event_weight = np.delete(event_weight, mask, axis = 0)

    return points, event_weight


# ---------------------------------------------------------------------------------------------------------------------------------
# Worker-side state. With the 'fork' start method (Linux) these globals are inherited from the parent for free;
# with 'spawn' (macOS, Windows) init_worker fills them in once per worker. Either way nothing large is pickled per task.
# ---------------------------------------------------------------------------------------------------------------------------------
Tree = None
worker_points = None
worker_truth_points = None

def init_worker(tree_path, point_filepath, weight_filepath, whattype, beta_value):
    global Tree, worker_points, worker_truth_points
    if calc_emds is None:
        configure_emd(beta_value)
    if worker_points is None:
        worker_points, _ = load_inputs(point_filepath, weight_filepath, whattype)
        worker_truth_points = worker_points[:, 0, :]
    if Tree is None:
        with open(tree_path, 'rb') as f:
            Tree = pickle.load(f)

def parallel_query(i, R):
    """Return, for negative event i, the indices of all events within EMD radius R, sorted by distance (nearest first)."""
    if i % 1000 == 0:
        print(i)
    try:
        sorted_neighbors = sorted(Tree.get_all_in_range(worker_points[i], R), key=lambda x: x[0])
        # the vp-tree hands back the neighbour's coordinates, not its index: recover the index by matching the leading particle
        return [np.where(np.all(np.abs(worker_truth_points - kin[0, :]) <= 0.001, axis=1))[0] for _, kin in sorted_neighbors]
    except RuntimeError:
        print(f'Error with event {i}')
        return [np.array([i])]


def _num_str(x):
    return f"{x:g}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vptree", type=str, required = True, help = "Path to vptree")
    parser.add_argument("--points", type=str, required = True, help = "Path to points file")
    parser.add_argument("--weights", type=str, required = True, help = "Path to weights file")
    parser.add_argument("--path", type=str, required = True, help = "Output filepath")
    parser.add_argument("--max_radius", type = float, required = True, help = "Max radius")
    parser.add_argument("--whattype", type = int, required = True, help = "0 = hard process, 1 = showered, 2 = hadronization")
    parser.add_argument("--beta", type=float, required=False, default=1, help = "Beta value for calculation of EMD; default is 1")
    parser.add_argument("--nproc", type=int, required=False, default=None, help = "Number of worker processes for vp-tree queries (default: CPUs available to this job)")
    args = parser.parse_args()

    global worker_points, worker_truth_points

    total_pc = time.perf_counter()
    total_pt = time.process_time()
    print('Starting data preprocessing...')

    configure_emd(args.beta)

    vptree_filepath = args.vptree
    max_radius = args.max_radius
    output_filepath = args.path
    whattype = args.whattype
    print("Max radius ", max_radius," beta ", beta)

    stage_strs = {0: "hp", 1: "ps", 2: "had"}
    if whattype not in stage_strs:
        raise ValueError(f"--whattype must be 0 (hard process), 1 (showered) or 2 (hadronization); got {whattype}")
    stage_str = stage_strs[whattype]

    num_cores = os.cpu_count()
    print(f"Number of CPU cores: {num_cores}")

    points, event_weight = load_inputs(args.points, args.weights, whattype)
    N = len(event_weight)
    neg_events = np.where(event_weight < 0)[0]

    # share the points with forked workers without pickling them per task
    worker_points = points
    worker_truth_points = points[:, 0, :]

    if args.nproc is not None:
        num_processes = args.nproc
    else:
        try:
            num_processes = len(os.sched_getaffinity(0))
        except AttributeError:
            num_processes = os.cpu_count()
    print("Using", num_processes, "worker processes")
    print("about to link vptree to weights ")

    start_pc = time.perf_counter()
    start_pt = time.process_time()
    with mp.Pool(processes=num_processes, initializer=init_worker,
                 initargs=(vptree_filepath, args.points, args.weights, whattype, args.beta)) as pool:
        results = pool.starmap(parallel_query, [(i, max_radius) for i in neg_events])
    print("success")
    pc = time.perf_counter()-start_pc
    pt = time.process_time() - start_pt
    print("time taken to link vptree: ", pc ," (perf count) ", pt ," (proc time)") 

    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    new_weight = np.copy(event_weight)
    cell_pop = []
    neg_cell_pop = []
    cell_radius = []
    print("Copy weights obj")
    print("Performing rw")
    start_pc = time.perf_counter()
    start_pt = time.process_time()
    print("N events ", N)
    print("Length of weights ", len(new_weight))
    n_empty = 0
    for i in range(N):

        if new_weight[i] < 0:

            cell_idx = [x[0] for x in results[np.where(neg_events == i)[0][0]] if len(x) > 0]
            max_cell = new_weight[cell_idx]
            cell_weight = 0
            abs_cell_weight = 0

            cumsum = np.cumsum(max_cell)
            #### protect against cells w/ no nearest neighbots
            if len(cumsum) < 1:
                n_empty += 1
                continue

            # Find the first index where cumulative sum becomes positive
            idx = np.argmax(cumsum > 0)  # position of first True        
            if np.any(cumsum > 0):
                num_elements = idx + 1  # +1 because indices start at 0
            else:
                num_elements = 1  # never becomes positive

            for j in range(num_elements):
                if cell_weight <=1E-5:
                    cell_weight += max_cell[j]
                    abs_cell_weight += np.abs(max_cell[j])
                else:
                    break

            if cell_weight <= 0:
                continue

            if num_elements > 1:
                cell_pop.append(num_elements)
                cell_radius.append(compute_emds(points[i], points[cell_idx[idx]]))
                neg_cell_pop.append(1+len(max_cell[0:idx][max_cell[0:idx] < 0]))

            new_weight[cell_idx[0:j+1]] = np.sum(cell_weight) / np.sum(abs_cell_weight) * np.abs(max_cell[0:j+1])

        if i % 10000 == 0:
            print(f'{i}')
    pc = time.perf_counter()-start_pc
    pt = time.process_time() - start_pt
    print("time taken to reweight ", pc ," (perf count) ", pt ," (proc time)") 
    if np.abs(new_weight.sum() - event_weight.sum()) < 0.1:
        print('Sum of weights agree', new_weight.sum(), event_weight.sum())
    else:
        warnings.warn("WARNING: SUM OF WEIGHTS BEFORE AND AFTER REWEIGHTING DO NOT MATCH", UserWarning)
    #-------------------------------------------------------------------------------------------------------------------------------------
    np.save(output_filepath, new_weight)
    print("Saved file to ", output_filepath)
    final_pc = time.perf_counter() - total_pc
    final_pt = time.process_time() - total_pt
    print("Total time taken: ", final_pc, " (pc) ", final_pt, " (pt)")

    max_radius_str = _num_str(max_radius)
    beta_str = _num_str(beta)

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cell_info_dir = os.path.join(repo_root, 'data', 'cell_info')
    os.makedirs(cell_info_dir, exist_ok=True)
    np.save(f'{cell_info_dir}/cell_radius_{stage_str}_{max_radius_str}gev_b{beta_str}.npy', cell_radius)
    np.save(f'{cell_info_dir}/cell_pop_{stage_str}_{max_radius_str}gev_b{beta_str}.npy', cell_pop)
    np.save(f'{cell_info_dir}/neg_cell_pop_{stage_str}_{max_radius_str}gev_b{beta_str}.npy', neg_cell_pop)
    print("Saved cell diagnostics to", cell_info_dir)


if __name__ == "__main__":
    main()
