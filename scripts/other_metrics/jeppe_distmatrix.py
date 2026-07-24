import numpy as np
import os
import h5py

#check number of cores                                                                                                                                                                                                           
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

N = 100000

event_weight = np.load('/oscar/data/mleblan6/rjain/ppzjj_100k/weight_100k.npy', mmap_mode='r')
#event_weight = np.load('/oscar/data/mleblan6/lhay/ttbar_100k/ttbar_weight_100k.npy', mmap_mode='r')
neg_events = np.where(event_weight < 0)[0]

#load in array and symmetrize
with h5py.File("/oscar/data/mleblan6/cell_resampling/distance_matrix_zjj_100k_typed_nu.h5", "r") as f:
    # inspect available datasets
    print(list(f.keys()))

    # load dataset into memory as a NumPy array
    original_dist2neg = np.array(f["distance_matrix"])

print('Distance matrix loaded')
i_lower = np.tril_indices_from(original_dist2neg, k=-1)
original_dist2neg[i_lower] = original_dist2neg.T[i_lower]

print('Distance matrix symmetrized')


#check if the symmetrization actually worked
counter = 0
for i in range(100):
    rand1 = np.random.randint(0, 100000)
    rand2 = np.random.randint(0, 100000)

    if original_dist2neg[rand1,rand2] == original_dist2neg[rand2,rand1]:
        counter+=1

    else:
        print(original_dist2neg[rand1,rand2])
        print(original_dist2neg[rand2,rand1])

print(counter)
if counter == 100:
    original_dist2neg = original_dist2neg[neg_events]
    np.save('/oscar/data/mleblan6/rjain/jeppe_new_v4/zjj_full_distmatrix.npy', original_dist2neg)

else:
    print('Symmetrization failed!')
