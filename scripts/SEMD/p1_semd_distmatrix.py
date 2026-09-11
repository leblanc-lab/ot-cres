import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def merge_arrays(A, B):
    # Unique B values and mapping indices
    unique_B, inv = np.unique(B, return_inverse=True)

    # Sum A values per unique B
    A_sum = np.zeros_like(unique_B, dtype=A.dtype)
    np.add.at(A_sum, inv, A)

    return A_sum, unique_B


def compute_spectral_representation(event, dtype=np.float64):
    pts = event[:, 0]
    etaphis = event[:, 1:]

    # Pairwise eta/phi differences
    diff = etaphis[:, None, :] - etaphis[None, :, :]
    
    # Euclidean distance
    distance_matrix = np.sqrt((diff ** 2).sum(axis=2))

    # Energy products
    ee = np.outer(pts, pts)

    # Flatten
    w = distance_matrix.ravel()
    s = ee.ravel()

    s, w = merge_arrays(s, w)

    mask = s > 0
    return w[mask].astype(dtype), s[mask].astype(dtype)

def cumulative_spectral_function(w,s):
    idx = np.argsort(w)
    W = w[idx]
    S = s[idx]
    S = np.cumsum(S)
    return W, S

def compute_SEMD(f_jumps, f_vals, g_jumps, g_vals):
    xmax = np.sqrt((2*np.pi)**2 + 9.8**2)
    
    pts = np.unique(np.concatenate((f_jumps, g_jumps, [xmax])))

    f_cum = np.r_[0, f_vals][np.searchsorted(f_jumps, pts, side="right")]
    g_cum = np.r_[0, g_vals][np.searchsorted(g_jumps, pts, side="right")]

    return np.sum(np.abs(f_cum[:-1] - g_cum[:-1]) * np.diff(pts))



numevents = 100000
event = np.load('/oscar/data/mleblan6/rjain/ppzjj_100k/showered_points.npy')[0:numevents]
weight = np.load('/oscar/data/mleblan6/rjain/ppzjj_100k/weight_100k.npy')[0:numevents]
neg_events = np.where(weight<0)[0]

all_S = []
all_W = []

print('Beginning cumulative spectral function generation')
for i in range(numevents):
    w, s = compute_spectral_representation(event = event[i])
    W, S = cumulative_spectral_function(w, s)

    all_S.append(S)
    all_W.append(W)

    if i % 1000 == 0:
        print(f'{i}/{numevents} completed')


print('All cumulative spectral functions generated! Now beginning integrals...')


distmatrix = np.zeros((len(neg_events), numevents))
def compute_row(i):
    Wi = all_W[neg_events[i]]
    Si = all_S[neg_events[i]]
    row = np.empty(numevents)

    for j in range(numevents):
        row[j] = compute_SEMD(Wi, Si, all_W[j], all_S[j])

    return i, row

results = Parallel(n_jobs=192)(
    delayed(compute_row)(i) for i in tqdm(range(len(neg_events)))
)

for i, row in results:
    distmatrix[i] = row


print(f'Max distance is: {np.max(distmatrix)}')
np.save('/oscar/data/mleblan6/rjain/p1_semd/showered_distmatrix.npy', distmatrix)