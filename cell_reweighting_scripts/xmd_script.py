import numpy as np
import os
import ot
import warnings

#check number of cores and compile specter
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

points = np.load('/oscar/data/mleblan6/rjain/ppzjj_100k/hadronization_points.npy')
weight = np.load('/oscar/data/mleblan6/rjain/ppzjj_100k/weight_100k.npy')
distmatrix = np.load('/oscar/data/mleblan6/rjain/had_100k_distmatrix/full_distmatrix.npy')

distmatrix /= np.max(distmatrix)

print(f'Full distance matrix loaded and normalized! Maximum is {np.max(distmatrix)}!')

def xmd_distance(q, w1_pos, w2_pos, pos_distmatrix):  
    #remove all positive events that are unaffected by reweighting
    mask0 = w1_pos != w2_pos

    w1_pos = w1_pos[mask0]
    w2_pos = w2_pos[mask0]
    pos_distmatrix = pos_distmatrix[mask0][:,mask0]

    #remove all negative events that have a weight of 0 after adding constant
    mask1 = w1_pos != 0
    mask2 = w2_pos != 0

    w1_pos = w1_pos[mask1]
    w2_pos = w2_pos[mask2]
    pos_distmatrix = pos_distmatrix[mask1][:, mask2]

    print(f'Sparse distance matrix has shape: {np.shape(pos_distmatrix)}')

    pos_xmd = ot.emd2(w1_pos/w1_pos.sum(), w2_pos/w2_pos.sum(), pos_distmatrix, numItermax = 10000000, numThreads = 150)
#    pos_xmd = ot.sinkhorn2(w1_pos/w1_pos.sum(), w2_pos/w2_pos.sum(), pos_distmatrix/biggest_emd, reg = 1e-1, warn = True)
    pos_xmd = pos_xmd * min(w1_pos.sum(), w2_pos.sum())
    
    distance = pos_xmd

    print('-----------------------------------------') 
    print(f'Total distance: {distance}')
    print('-----------------------------------------') 
    return(distance)

const = np.abs(min(weight))
print(f'Constant is {const}')


radii = np.logspace(2,4,50)

reweights = np.load(f'/oscar/data/mleblan6/rjain/semd_reweight/hardprocess_reweights.npy')

semd_xmd_distances = []

print('############################################################')
print('Now starting SEMD XMDs')
print('############################################################')

for i in range(len(radii)):
    print('------------------------------------------------------------')
    print(f'Now starting: max radius = {radii[i]}')
    
    print('------------------------------------------------------------')
    if np.min(reweights[i]+const < 0):
        warnings.warn("Reweight array has negative values! XMD will not work", UserWarning)
    semd_xmd_distances.append(xmd_distance(points, (reweights[i]+const), (weight+const), pos_distmatrix = distmatrix))
np.save('hp_semd_XMD', semd_xmd_distances)
