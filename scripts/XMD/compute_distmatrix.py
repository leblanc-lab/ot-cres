#temp
import numpy as np
import vptree
import os
import wasserstein
import multiprocessing as mp
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--index", type = int, default = None, required = False, help = "Which portion of distmatrix is being computed")
parser.add_argument("--points", type = str, required = True, help = "Path of hadronization points file to run over")
parser.add_argument("--output", type = str, required = True, help = "Path of output file (.npy)")
args = parser.parse_args()



#check number of cores and compile specter
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

points = np.load(args.points)

if args.index is None:
    print("Running over all points in sample")
    trialnum = 0
    N = len(points)
else:
    trialnum = args.index
    N = 10000
    print(f"Running over points {trialnum}*{N}:({1+trialnum})*{N}")

points_small = points[trialnum*N:(1+trialnum)*N]


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


#distance from all events to negative event (parallelized)
def hadron_calc_emds_multiproc(i, j, points1, points2):
    if i % 100 == 0 and j == 0:
        print(i)
    try:
        return compute_emds2(points1[i], points2[j])
        
    except RuntimeError as e:
        print(f"Error with EMD calculation between event {i} and {j}: {str(e)}")
        return 1000


start_pc = time.perf_counter()
start_pt = time.process_time()

with mp.Pool(processes=mp.cpu_count()) as pool:
    results = pool.starmap(
        hadron_calc_emds_multiproc, 
        [(i, j, points_small, points) for i in range(N) for j in range(len(points))]
    )
    
pc = time.perf_counter()-start_pc
pt = time.process_time() - start_pt
print("time taken to make dist matrix ", pc ," (perf count) ", pt ," (proc time)") 

full_distmatrix = np.array(results).reshape(N,len(points))

np.save(args.output, full_distmatrix)