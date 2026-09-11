import numpy as np
import fastjet as fj
import uproot
import awkward as ak
import vector

vector.register_awkward()

rem_leptons = True


file = uproot.open('/oscar/data/mleblan6/lhay/ttbar_100k/ttbar_NLO_100k.root')
print('Opened root file')
tree = file['Events']
particle_eta_cut = 4.9
particle_pt_cut = 0.0
N=100000
data = tree.arrays(["Particle_energy", "Particle_px","Particle_py", "Particle_pz", "Particle_status", "Particle_mass", "Particle_pid"])
print('Created initial data array')
print(data)


	#Determine size of array
# maximum = 0

# for i in range(N):
#     	if len(np.where(data[i]['Particle_status'] == 1)[0]) > maximum:
#         	maximum = len(np.where(data[i]['Particle_status'] == 1)[0])

# print(f'Max number of particles is: {maximum}')

px     = data["Particle_px"]
py     = data["Particle_py"]
pz     = data["Particle_pz"]
E      = data["Particle_energy"]
mass   = data["Particle_mass"]
pid    = data["Particle_pid"]
status = data["Particle_status"]

pt  = np.sqrt(px**2 + py**2)
eta = np.arctanh(pz / np.sqrt(px**2 + py**2 + pz**2))
phi = np.arctan2(py, px)

print("doing things awkwardly")

abs_pid = np.abs(pid)

is_final = status == 1
is_lepton = (abs_pid == 11) | (abs_pid == 13)
is_neutrino = (abs_pid == 12) | (abs_pid == 14) | (abs_pid == 16)

is_b_hadron = (
    ((abs_pid // 100) % 10 == 5)
    | ((abs_pid // 1000) % 10 == 5)
)

jet_particle_mask = (
    is_final
    & ~is_lepton
    & ~is_neutrino
    & (pt >= particle_pt_cut)
    & (abs(eta) <= particle_eta_cut)
)

jet_particles= ak.zip(
    {
        "pt": pt[jet_particle_mask],
        "eta": eta[jet_particle_mask],
        "phi": phi[jet_particle_mask],
        "M": mass[jet_particle_mask],
    },
    with_name="Momentum4D",
)

lep_mask = is_final & is_lepton & (pt > 0)

leptons = ak.zip(
    {
        "pt": pt[lep_mask],
        "eta": eta[lep_mask],
        "phi": phi[lep_mask],
        "M": mass[lep_mask],
        "pdgId": pid[lep_mask],
    },
    with_name="Momentum4D",
)
ak.to_parquet(leptons, 'ttbar_NLO_100k_leptons.parquet')

b_mask = is_b_hadron

b_hadrons = ak.zip(
    {
        "pt": pt[b_mask],
        "eta": eta[b_mask],
        "phi": phi[b_mask],
        "M": mass[b_mask],
        "pdgId": pid[b_mask],
    },
    with_name="Momentum4D",
)

ak.to_parquet(b_hadrons, 'ttbar_NLO_100k_bhadrons.parquet')

#--------------------------------------------------------------------
#JET CLUSTERING STUFF BELOW
#--------------------------------------------------------------------



R = 0.4

akt_jetdef = fj.JetDefinition(fj.antikt_algorithm, R)


#split up the sample
#do jet clustering
jet_pt_cutoff = 20
jet_eta_cutoff = 4.5

alljets = []

for j in range(N):

    akt_cluster = fj.ClusterSequence(jet_particles[j], akt_jetdef);
    akt_jets = akt_cluster.inclusive_jets()

    #create n_jets observable
    good_ind = []
    for k in range(len(akt_jets)):
        jpx, jpy, jpz = akt_jets[k]['px'], akt_jets[k]['py'], akt_jets[k]['pz']
        jet_pt_k = np.sqrt(jpx**2 + jpy**2)
        jet_eta_k = np.arctanh(jpz / np.sqrt(jpx**2 + jpy**2 + jpz**2))
        if (jet_pt_k > jet_pt_cutoff) & (np.abs(jet_eta_k) < jet_eta_cutoff):
            good_ind.append(k)

    akt_jets = akt_jets[good_ind]

    alljets.append(akt_jets)

    if j % 5000 == 0:
        print(f'Completed Events: {j} / {N}')


print('Completed all clustering! Now starting coordinate transformation!')

alljets = ak.Array(alljets)

jet_pt = np.sqrt(alljets.px**2 + alljets.py**2)
jet_eta = np.arctanh(alljets.pz / np.sqrt(alljets.px**2 + alljets.py**2 + alljets.pz**2))
jet_phi = np.arctan2(alljets.py, alljets.px)

m2 = (alljets.E*alljets.E) - (alljets.px**2+alljets.py**2+alljets.pz**2)
jet_mass  = np.sqrt(np.maximum(m2, 0.0))

alljets_coords = ak.zip({'pt': jet_pt, 'eta': jet_eta, 'phi': jet_phi, 'mass':jet_mass}, with_name="Momentum4D")

print('Finished coordinate transformation!') 

ak.to_parquet(alljets_coords, '/oscar/data/mleblan6/lhay/ttbar_NLO_100k_eta4p5_pt20.parquet')
