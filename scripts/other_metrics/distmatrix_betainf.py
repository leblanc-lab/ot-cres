import numpy as np
import os 
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--whattype", type = int, required = True, help = "0 = hard process, 1 = showered, 2 = hadronization")
args = parser.parse_args()


whattype = args.whattype

if whattype == 0:
    points = np.load('/users/rrjain/reweight_100k/hardprocess_points.npy')
elif whattype == 1:
    points = np.load('/users/rrjain/reweight_100k/showered_points.npy')
elif whattype == 2:
    points = np.load('/users/rrjain/reweight_100k/hadronization_points.npy')

weight = np.load('/oscar/data/mleblan6/rjain/ppzjj_100k/weight_100k.npy')

neg_events = np.where(weight<0)[0]
pt_arr = points[:,:,0]


def calc_dist_multiproc(i, j, pt_arr):
    if i % 1000 == 0 and j == 9990:
        print(i)
    try:
        return np.abs(pt_arr[i,:].sum() - pt_arr[j,:].sum())
    except RuntimeError as e:
        print(f"Error with EMD calculation between event {i} and {j}: {str(e)}")
        return 1000

with mp.Pool(processes=180) as pool:
    results = pool.starmap(
        calc_dist_multiproc, 
        [(i, j, pt_arr) for i in neg_events for j in range(100000)]
    )

dist2neg = np.array(results).reshape(len(neg_events), 100000)

if whattype == 0:
    np.save('/oscar/data/mleblan6/rjain/betainf/hardprocess_dist2neg', dist2neg)
elif whattype == 1:
    np.save('/oscar/data/mleblan6/rjain/betainf/showered_dist2neg', dist2neg)
elif whattype == 2:
    np.save('/oscar/data/mleblan6/rjain/betainf/hadronization_dist2neg', dist2neg)

print('Finished')