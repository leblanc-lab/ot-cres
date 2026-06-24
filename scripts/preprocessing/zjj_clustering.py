import numpy as np
import fastjet as fj
import uproot
import awkward as ak
import vector



file = uproot.open("/oscar/data/mleblan6/rjain/ppzjj_100k/ppzjj_NLO_100k.root")
print('Opened root file')
tree = file['Events']
print(tree.keys())
no_lep = True

N = 100000
particle_y_cut = 4.9
particle_pt_cut = 0.1

data = tree.arrays(["Particle_energy", "Particle_px","Particle_py", "Particle_pz", "Particle_status", "Particle_mass", "Particle_pid"])
print('Created initial data array')


	#Determine size of array
maximum = 0

for i in range(N):
    	if len(np.where(data[i]['Particle_status'] == 1)[0]) > maximum:
        	maximum = len(np.where(data[i]['Particle_status'] == 1)[0])

print(f'Max number of particles is: {maximum}')


	#Make and save main coords of particles
main_coords = np.zeros((N,maximum,4))
for i in range(N):
    if no_lep:
        pid = data[i]['Particle_pid']
        # e, nu_e, mu, nu_mu, tau, nu_tau
        lep_mask = ~((pid == 11) | (pid == 12) | (pid == 13) | (pid == 14) | (pid == 15) | (pid == 16))
        mask = (data[i]['Particle_status'] == 1) & lep_mask
    else:
        mask = (data[i]['Particle_status'] == 1)
    n_selected = ak.sum(mask)
    if i % 10000 == 0:
        print(f"Event {i}: selected particles = {n_selected}/{len(pid)}")
    temp_len = int(ak.sum(mask))
    px = np.array(data[i]['Particle_px'][mask]).astype(np.float32)
    py = np.array(data[i]['Particle_py'][mask]).astype(np.float32)
    pz = np.array(data[i]['Particle_pz'][mask]).astype(np.float32)
    E = np.array(data[i]['Particle_energy'][mask]).astype(np.float32)
    mass = np.array(data[i]['Particle_mass'][mask]).astype(np.float32)

    main_coords[i, 0:temp_len, 0] = np.sqrt(px**2 + py**2)
    main_coords[i, 0:temp_len, 1] = 0.5 * np.log((E + pz) / (E - pz))
    main_coords[i, 0:temp_len, 2] = np.arctan2(py, px)
    main_coords[i, 0:temp_len, 3] = mass

    for j in range(maximum):
        if main_coords[i, j, 0] < particle_pt_cut:
            main_coords[i, j, :] = [0, 0, 0, 0]

        elif abs(main_coords[i, j, 1]) > particle_y_cut:
            main_coords[i, j, :] = [0, 0, 0, 0]

    if i % 10000 == 0:
        print(f'Generated coordinates for {i} / {N} events')

print('All final state particle coordinates processesed!')



sample = main_coords

alljets = []
#--------------------------------------------------------------------
#JET CLUSTERING STUFF BELOW
#--------------------------------------------------------------------



R = 0.4

akt_jetdef = fj.JetDefinition(fj.antikt_algorithm, R)


#split up the sample

hadronization_EMD_pt = sample[:,:,0]
hadronization_EMD_etaphi = sample[:,:,1:3]
hadronization_p_mass = sample[:,:,3]



#do jet clustering
jet_pt_cutoff = 20
jet_eta_cutoff = 4.5

vector.register_awkward()



#make awk arrays for all particles
mask = hadronization_EMD_pt != 0

jets = ak.zip({
    "pt": hadronization_EMD_pt[mask],
    "eta": hadronization_EMD_etaphi[mask, 0],
    "phi": hadronization_EMD_etaphi[mask, 1],
    "M": hadronization_p_mass[mask]
}, with_name="Momentum4D")

awk_particle_array = ak.unflatten(jets, ak.sum(mask, axis=1))


for j in range(N):

    akt_cluster = fj.ClusterSequence(awk_particle_array[j], akt_jetdef);
    akt_jets = akt_cluster.inclusive_jets()

    #create n_jets observable
    good_ind = []
    for k in range(len(akt_jets)):
        if (np.sqrt(akt_jets[k]['px']**2 + akt_jets[k]['py']**2) > jet_pt_cutoff) & (np.abs(np.arctanh(akt_jets[k]['pz'] / akt_jets[k]['E']))<jet_eta_cutoff):
            good_ind.append(k)

    akt_jets = akt_jets[good_ind]

    alljets.append(akt_jets)

    if j % 5000 == 0:
        print(f'Completed Events: {j} / {N}')


print('Completed all clustering! Now starting coordinate transformation!')

alljets = ak.Array(alljets)

jet_pt = np.sqrt(alljets.px**2 + alljets.py**2)
jet_eta = np.arctanh(alljets.pz / alljets.E)
jet_phi = np.arctan2(alljets.py, alljets.px)

m2 = (alljets.E*alljets.E) - (alljets.px**2+alljets.py**2+alljets.pz**2)
jet_mass  = np.sqrt(np.maximum(m2, 0.0))

alljets_coords = ak.zip({'pt': jet_pt, 'eta': jet_eta, 'phi': jet_phi, 'mass':jet_mass}, with_name="Momentum4D")

print('Finished coordinate transformation!') 
if no_lep:
    ak.to_parquet(alljets_coords, '/oscar/data/mleblan6/lhay/zjj_NLO_100k_eta4p5_NOLEP.parquet')
else:
    ak.to_parquet(alljets_coords, '/oscar/data/mleblan6/lhay/zjj_NLO_100k_eta4p5.parquet')
