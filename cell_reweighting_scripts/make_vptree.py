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
parser.add_argument("--point_path", type=str, required = True, help = "Input file of points (had, parton shower, or hard process .npy file)")
parser.add_argument("--beta", type=float, required=False, default=1, help = "Beta value for calculation of EMD; default is 1")
# parser.add_argument("--whattype", type = int, required = True, help = "0 = hard process, 1 = showered, 2 = hadronization")

args = parser.parse_args()

# whattype = args.whattype
outpath = args.path
print("Got args")
#parameters of EMD calculation
max_dist = np.sqrt(9.8**2 + (2*np.pi)**2)
print("define max dist ", max_dist)
calc_emds = wasserstein.EMDYPhi(R=max_dist, 
                                        beta=args.beta,
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

print("Computed emds")

points = np.load(args.point_path)

print("Loaded points -- next make vptree")

tree = vptree.VPTree(points, compute_emds)

print("Made vptree -- saving now")

with open(outpath, "wb") as f:
    pickle.dump(tree, f)
