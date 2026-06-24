import numpy as np
import pyhepmc

#read hepmc file of mc data and store event weights
filepath = '/eos/user/r/rijain/ttbar_NLO_100k/Events/run_01/ttbar_NLO_100k.hepmc'
xsec = np.array([])

with pyhepmc.open(filepath) as f:
    for i in f:
        xsec = np.append(xsec, i.weights[0])

print(f'There are {len(np.where(xsec < 0)[0])} negative weighted events out of {len(xsec)} events')

np.save('ttbar_weight_100k.npy', xsec)
