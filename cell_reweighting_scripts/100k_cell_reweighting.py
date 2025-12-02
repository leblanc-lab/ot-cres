#import useful stuff
import pickle 
import numpy as np
import uproot
import wasserstein
import os
import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
import warnings

#import arguments

parser = argparse.ArgumentParser()


parser.add_argument("--vptree", type=str, required = True, help = "Path to vptree")
parser.add_argument("--path", type=str, required = True, help = "Output filepath")
parser.add_argument("--max_radius", type = float, required = True, help = "Max radius")
parser.add_argument("--whattype", type = int, required = True, help = "0 = hard process, 1 = showered, 2 = hadronization")

args = parser.parse_args()



print('Starting data preprocessing...')
# #-----------------------------------------------------------------------------------------------------------------------------------------------------------
#compute EMDs 
#parameters of EMD calculation
max_dist = np.sqrt(9.8**2 + (2*np.pi)**2)
calc_emds = wasserstein.EMDYPhi(R=max_dist, 
                                        beta=1,
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


def compute_emds(points1, points2):
    pts1 = points1[:,0]
    pts2 = points2[:,0]
    etaphi1 = points1[:,1:3]
    etaphi2 = points2[:,1:3]

    return calc_emds(pts1, etaphi1, pts2, etaphi2)

# #----------------------------------------------------------------------------------------------------------------------------------
#load in distance matrices and remove repeated distances and zeros

vptree_filepath = args.vptree

#import vptree
with open(vptree_filepath, "rb") as f:
    test = pickle.load(f)

max_radius = args.max_radius
output_filepath = args.path
whattype = args.whattype

#check number of cores
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

N = 100000 #number of events

if whattype == 0:
    points = np.load('/oscar/data/mleblan6/ppzjj_100k/hardprocess_points.npy')
    
elif whattype == 1: 
    points = np.load('/oscar/data/mleblan6/ppzjj_100k/showered_points.npy')
    
elif whattype == 2:
    points = np.load('/oscar/data/mleblan6/ppzjj_100k/hadronization_points.npy')


event_weight = np.load('/oscar/data/mleblan6/ppzjj_100k/weight_100k.npy') * 1.455 * 10**4

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
with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(vptree_filepath,)) as pool:
    results = pool.starmap(parallel_query, [(i, points, max_radius, truth_points) for i in neg_events])

#-----------------------------------------------------------------------------------------------------------------------------------------------------
new_weight = np.copy(event_weight)
event_num = []
cell_pop = []
neg_cell_pop = []
cell_radius = []

for i in range(N):
            
    if new_weight[i] < 0:


        cell_idx = [x[0] for x in results[np.where(neg_events == i)[0][0]]]
        max_cell = new_weight[cell_idx]
        cell_weight = 0
        abs_cell_weight = 0

        cumsum = np.cumsum(max_cell)

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

        #print(i, cell_idx)
        if num_elements > 1:
            cell_pop.append(num_elements)
            cell_radius.append(compute_emds(points[i], points[cell_idx[idx]]))
            neg_cell_pop.append(1+len(max_cell[0:idx][max_cell[0:idx] < 0]))
            
        new_weight[cell_idx[0:j+1]] = np.sum(cell_weight) / np.sum(abs_cell_weight) * np.abs(max_cell[0:j+1])

    if i % 10000 == 0:
        print(f'{i}')

if np.abs(new_weight.sum() - event_weight.sum()) < 0.1:
    print('Sum of weights agree', new_weight.sum(), event_weight.sum())
else:
    warnings.warn("WARNING: SUM OF WEIGHTS BEFORE AND AFTER REWEIGHTING DO NOT MATCH", UserWarning)
#-------------------------------------------------------------------------------------------------------------------------------------
np.save(output_filepath, new_weight)

bins = np.linspace(0, int(np.ceil(max_radius)), int(np.ceil(max_radius)) + 1)

plt.figure()
plt.hist(cell_radius, bins = bins);
plt.yscale('log')
plt.xlabel('Radius [GeV]')
plt.ylabel('Frequency');
if whattype == 2:
	plt.title('Hadronization Reweight Cell Radius')
    plt.savefig(f'cell_radius_had_{int(max_radius)}gev.png')
elif whattype == 1:
	plt.title('Showered Reweight Cell Radius')
    plt.savefig(f'cell_radius_sho_{int(max_radius)}gev.png')
elif whattype == 0:
	plt.title('Hard Process Cell Radius')
    plt.savefig(f'cell_radius_hp_{int(max_radius)}gev.png')


plt.figure()
plt.hist(cell_pop, bins = 25);
plt.yscale('log')
plt.title(f'Number of Events in Cell (R = {max_radius} GeV)')
plt.xlabel('#')
plt.ylabel('Frequency');
if whattype == 2:
    plt.savefig(f'cell_pop_had_{int(max_radius)}gev.png')
elif whattype == 1:
    plt.savefig(f'cell_pop_sho_{int(max_radius)}gev.png')
elif whattype == 0:
    plt.savefig(f'cell_pop_hp_{int(max_radius)}gev.png')


plt.figure()
plt.hist(neg_cell_pop, bins = 25);
plt.yscale('log')
plt.title(f'Number of Negative Events in Cell (R = {max_radius} GeV)')
plt.xlabel('#')
plt.ylabel('Frequency');
plt.savefig(f'neg_cell_pop_{whattype}_{max_radius}gev.png')
if whattype == 2:
    plt.savefig(f'neg_cell_pop_had_{int(max_radius)}gev.png')
elif whattype == 1:
    plt.savefig(f'neg_cell_pop_sho_{int(max_radius)}gev.png')
elif whattype == 0:
    plt.savefig(f'neg_cell_pop_hp_{int(max_radius)}gev.png')

