import pickle 
import numpy as np
import uproot
import wasserstein
import os
import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
import warnings
import time

#import arguments

parser = argparse.ArgumentParser()


parser.add_argument("--vptree", type=str, required = True, help = "Path to vptree")
parser.add_argument("--points", type=str, required = True, help = "Path to points file")
parser.add_argument("--weights", type=str, required = True, help = "Path to weights file")
parser.add_argument("--path", type=str, required = True, help = "Output filepath")
parser.add_argument("--max_radius", type = float, required = True, help = "Max radius")
parser.add_argument("--whattype", type = int, required = True, help = "0 = hard process, 1 = showered, 2 = hadronization")
parser.add_argument("--beta", type=float, required=False, default=1, help = "Beta value for calculation of EMD; default is 1")
# parser.add_argument("--niter", type = int, default=100000, help = "Max number of iterations (i.e. events). Must match vptree.")

args = parser.parse_args()
beta = args.beta

total_pc = time.perf_counter()
total_pt = time.process_time()
print('Starting data preprocessing...')
# #-----------------------------------------------------------------------------------------------------------------------------------------------------------
#compute EMDs 
#parameters of EMD calculation
max_dist = np.sqrt(9.8**2 + (2*np.pi)**2)
calc_emds = wasserstein.EMDYPhi(R=max_dist, 
                                        beta=beta,
                                        norm=False,
                                        #num_threads=-1,
                                        #print_every=1000,
                                        #verbose=0,
                                        # request_mode=False,
                                        #store_sym_emds_raw=True,
                                        #throw_on_error=False,
                                        # omp_dynamic_chunksize=10,
                                        n_iter_max=100000,
                                        #epsilon_large_factor=1000.0,
                                        #epsilon_small_factor=1.0,
                                        dtype='float64')

if beta <= 1:
    def compute_emds(points1, points2):
        pts1 = points1[:, 0]
        pts2 = points2[:, 0]
    
        etaphi1 = points1[:, 1:3]
        etaphi2 = points2[:, 1:3]
    
        if np.all(pts1 == 0):
            pts1[0] += 1e-5
    
        if np.all(pts2 == 0):
            pts2[0] += 1e-5
    
        try: 
            return calc_emds(pts1, etaphi1, pts2, etaphi2)
    
        except RuntimeError: #sometimes the EMD optimization fails to converge 
    
            pts1[-1] += 1e-5
            pts2[-1] += 1e-5
            return calc_emds(pts1, etaphi1, pts2, etaphi2)
elif beta > 1.0:
    def compute_emds(points1, points2):
        pts1 = points1[:, 0]
        pts2 = points2[:, 0]
    
        ht1 = pts1.sum()
        ht2 = pts2.sum()
    
        dE = np.abs(ht1 - ht2)
    
        etaphi1 = points1[:, 1:3]
        etaphi2 = points2[:, 1:3]
    
        if np.all(pts1 == 0):
            pts1[0] += 1e-5
    
        if np.all(pts2 == 0):
            pts2[0] += 1e-5
    
        try:
            return (calc_emds(pts1, etaphi1, pts2, etaphi2) - dE)**(1.0/beta) + dE
    
        except Exception as e:
            etaphi1[0,0] += 1e-5
            etaphi2[0,0] += 1e-5
            
            return (calc_emds(pts1, etaphi1, pts2, etaphi2) -dE)**(1.0/beta) + dE

# #----------------------------------------------------------------------------------------------------------------------------------
#load in distance matrices and remove repeated distances and zeros

vptree_filepath = args.vptree

#import vptree
with open(vptree_filepath, "rb") as f:
    test = pickle.load(f)

max_radius = args.max_radius
output_filepath = args.path
whattype = args.whattype
point_filepath = args.points
weight_filepath = args.weights
print("Max radius ", max_radius," beta ", beta)
if "had" in vptree_filepath or whattype==2:
    stage_str = "had"
elif "ps" in vptree_filepath or whattype==1:
    stage_str = "ps"
elif "hp" in vptree_filepath or "hs" in vptree_path or whattype==0:
    stage_str = "hp"
else:
    print("inconsistent stage given")

#check number of cores
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

if whattype == 0:
    points = np.load(point_filepath)
    event_weight = np.load(weight_filepath)
    
    mask = np.where(np.all(points == 0, axis=(1, 2)))[0]
    points = np.delete(points, mask, axis = 0)
    event_weight = np.delete(event_weight, mask, axis = 0)

    N = len(event_weight)

    
    
else:
    points = np.load(point_filepath)
    N = 100000 #number of events
    event_weight = np.load(weight_filepath)
    event_weight = event_weight[:N]
neg_events = np.where(event_weight < 0)[0]
Tree = None

def init_worker(tree_path):
    global Tree
    with open(tree_path, 'rb') as f:
        Tree = pickle.load(f)
        
def parallel_query(i, points, R, truth_points):
    if i % 1000 == 0:
        print(i)
    try:
        query = points[i]
        sorted_neighbors = sorted(Tree.get_all_in_range(query, R), key=lambda x: x[0])
        kinematics = [row[1] for row in sorted_neighbors]
        prox_array = []
        prox_array = [np.where(np.all(np.abs(truth_points - kin[0, :]) <= 0.001, axis=1))[0] for kin in kinematics]
        return prox_array

    except RuntimeError as e:
        print(f'Error with event {i}')
        return [np.array([i])]


truth_points = points[:,0,:]

num_processes = 180
print("about to link vptree to weights ")

start_pc = time.perf_counter()
start_pt = time.process_time()
with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(vptree_filepath,)) as pool:
    results = pool.starmap(parallel_query, [(i, points, max_radius, truth_points) for i in neg_events])
print("success")
pc = time.perf_counter()-start_pc
pt = time.process_time() - start_pt
print("time taken to link vptree: ", pc ," (perf count) ", pt ," (proc time)") 
#-----------------------------------------------------------------------------------------------------------------------------------------------------
new_weight = np.copy(event_weight)
event_num = []
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
        # print("Cells with negative weights ", cell_idx)
        max_cell = new_weight[cell_idx]
        cell_weight = 0
        abs_cell_weight = 0

        cumsum = np.cumsum(max_cell)
        # print("cumulative sum ", cumsum)
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


if "." in str(max_radius):
    max_radius_str = str(max_radius).replace(".", "p")
else:
    max_radius_str = str(max_radius)

np.save(f'../data/cell_info/cell_radius_{stage_str}_{max_radius_str}gev_b{int(beta)}.npy', cell_radius)
np.save(f'../data/cell_info/cell_pop_{stage_str}_{max_radius_str}gev_b{int(beta)}.png', cell_pop)
np.save(f'../data/cell_info/neg_cell_pop_{stage_str}_{max_radius_str}gev_b{int(beta)}.png', neg_cell_pop)