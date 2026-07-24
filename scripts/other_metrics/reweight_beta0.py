import numpy as np

path = '/oscar/data/mleblan6/rjain/'
points = np.load(f'{path}ppzjj_100k/hadronization_points.npy')
weight = np.load(f'{path}ppzjj_100k/weight_100k.npy')

ht = [np.sum(points[i,:,0]) for i in range(len(points))]
ht = np.array(ht)

def beta0_reweight(radius, ht_arr, weight_arr):
    N = len(ht_arr)

    order = sorted(range(len(ht_arr)), key=lambda i: ht_arr[i])

    sorted_ht = [ht_arr[i] for i in order]
    reordered_weight = np.array([weight_arr[i] for i in order])

    for i in range(N):
        if reordered_weight[i] < 0 and sorted_ht[i] < radius:
            
            cell_idx = np.where(sorted_ht <= sorted_ht[i])[0]
        
            cell_sum = np.sum(reordered_weight[cell_idx])
            abs_cell_sum = np.sum(np.abs(reordered_weight[cell_idx]))

            if cell_sum > 0.01:
                reordered_weight[cell_idx] = np.abs(reordered_weight[cell_idx]) * cell_sum / abs_cell_sum

    reweight = [None] * len(reordered_weight)

    for new_idx, old_idx in enumerate(order):
        reweight[old_idx] = reordered_weight[new_idx]

    if np.abs(np.array(reweight).sum() - weight_arr.sum()) > 0.01:
        print('WARNING: SUM OF WEIGHTS DONT MATCH')


    print(f'{1 - np.sum(np.array(reweight) < 0)/37581} negative weights reweighted at max radius: {radius} GeV')
    
    return np.array(reweight)


radii = np.logspace(np.log10(50), np.log10(350), 50)
reweights = np.zeros((50, len(ht)))

for i in range(50):
    reweights[i] = beta0_reweight(radii[i], ht, weight)

np.save('had_beta0_reweights.npy', reweights)