#import useful stuff
import pickle 
import numpy as np
import uproot
import wasserstein
import os
import argparse
import vptree

#check number of cores and compile specter
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

#import arguments

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required = True, help = "Output filepath. WARNING THIS HAS TO END IN .pkl FORMAT")
parser.add_argument("--whattype", type = int, required = True, help = "0 = hard process, 1 = showered, 2 = hadronization")

args = parser.parse_args()

whattype = args.whattype
outpath = args.path


#parameters of EMD calculation
max_dist = np.sqrt(9.8**2 + (2*np.pi)**2)

calc_emds = wasserstein.EMDYPhi(R=max_dist, 
                                        beta=1,
                                        norm=True,
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
    pts1 = points1[:, 0]
    pts2 = points2[:, 0]

    etaphi1 = points1[:, 1:3]
    etaphi2 = points2[:, 1:3]

    if np.all(pts1 == 0):
        pts1[0] += 1e-5

    if np.all(pts2 == 0):
        pts2[0] += 1e-5

    return calc_emds(pts1, etaphi1, pts2, etaphi2)


if whattype == 0:
    points = np.load('/oscar/data/mleblan6/rjain/ppzjj_100k/hardprocess_points.npy')
elif whattype == 1:
    points = np.load('/oscar/data/mleblan6/rjain/ppzjj_100k/showered_points.npy')
elif whattype == 2:
    points = np.load('/oscar/data/mleblan6/rjain/ppzjj_100k/hadronization_points.npy')

N = 100000

tree = vptree.VPTree(points, compute_emds)

with open(outpath, "wb") as f:
    pickle.dump(tree, f)