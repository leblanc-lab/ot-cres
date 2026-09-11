# Import useful stuff
import uproot
import numpy as np
import wasserstein
import os
import argparse
import pyhepmc
from pyspecter.SPECTER import SPECTER
#from pyspecter.Observables import Observable
import time
import contextlib
import os
import jax
import jax.numpy as jnp

#check number of cores
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

#import arguments
parser = argparse.ArgumentParser()

parser.add_argument("--filepath", type=str, required=True, help="Path to the root input file of particle data")
parser.add_argument("--weight_filepath", type=str, required=True, help="Path to the hepmc input file of weights")
parser.add_argument("--N", type=int, required=True,help='How many samples to analyze')
parser.add_argument("-hp","--hardprocess",action='store_true', help='Calculate hard process SEMDS')
parser.add_argument("-sh","--showered",action='store_true', help='Calculate showered SEMDS')
parser.add_argument("-had","--hadronization",action='store_true', help='Calculate hadronization SEMDS')
args = parser.parse_args()



# check gpu is working

print("JAX DEVICES:", jax.devices())
jax.config.update("jax_enable_x64", True)
specter = SPECTER(compile = True)

#----------------------------------------------------------------------------------------------------------------------------
#define important functions

#function to apply cuts to data

def apply_cuts(data, etacut, ptcut, event_num):
    
    num_particles = len(data[event_num]["Particle_px"])
    
    four_momenta = np.zeros((num_particles, 4))
    
    four_momenta[:,0] = data[event_num]["Particle_energy"]
    four_momenta[:,1] = data[event_num]["Particle_px"]
    four_momenta[:,2] = data[event_num]["Particle_py"]
    four_momenta[:,3] = data[event_num]["Particle_pz"]
    
    status = data[event_num]["Particle_status"]
    pid = data[event_num]["Particle_pid"]
    daughter1 = data[event_num]["Particle_d1"]
    daughter2 = data[event_num]["Particle_d2"]
    
    mass = data[event_num]["Particle_mass"]
    
    num = 0
   
    removal_idx = np.array([])
    
    for i in range(num_particles):
        if four_momenta[i,1]**2 + four_momenta[i,2]**2 < ptcut**2:
#            print(i)
            removal_idx = np.append(removal_idx, int(i))
            num = num + 1
        
        elif abs(np.arctanh(four_momenta[i,3]/np.sqrt(four_momenta[i,1]**2 + four_momenta[i,2]**2 + four_momenta[i,3]**2))) > etacut:
            removal_idx = np.append(removal_idx, int(i))
            num = num + 1
            
            
    removal_idx = removal_idx.astype(int)
    four_momenta = np.delete(four_momenta, removal_idx, axis = 0)
    status = np.delete(status, removal_idx, axis = 0)
    pid = np.delete(pid, removal_idx, axis = 0)
    daughter1 = np.delete(daughter1, removal_idx, axis = 0)
    daughter2 = np.delete(daughter2, removal_idx, axis = 0)
    mass = np.delete(mass, removal_idx, axis = 0)
     
    
    return four_momenta, status, pid, daughter1, daughter2, mass, f"{num} particles removed out of {num_particles}"




#function to transform into (pT, eta, phi) coordinates

def coord_transform(data):
    
    num_particles = len(data[:,1])
    
    new_coords = np.zeros((num_particles, 3))
    
    new_coords[:,0] = np.sqrt(data[:,1]**2 + data[:,2]**2)
    new_coords[:,1] = np.arctanh(data[:,3]/np.sqrt(data[:,1]**2 + data[:,2]**2 + data[:,3]**2))
    new_coords[:,2] = np.arctan2(data[:,2],data[:,1]) 
            
    
    return new_coords


#----------------------------------------------------------------------------------------------------------------------------
#import data
filepath = args.filepath
file = uproot.open(filepath)
tree = file["Events"]


#define cut parameters

particle_eta_cut = 4.9
particle_pt_cut = 0.1

N = args.N  #number of events to analyze

#create data of 4 momenta
data = tree.arrays(["Particle_energy", "Particle_px","Particle_py", "Particle_pz", "Particle_status", "Particle_pid", 
                    "Particle_d1", "Particle_d2", "Particle_mass"])
data = data[0:N]


#which event has the most particles? This will set the size of higher dimensional space
maximum = 0

for i in range(N):
    four_mom = apply_cuts(data, particle_eta_cut, particle_pt_cut, i)[0]
    if len(four_mom[:, 0]) > maximum:
        maximum = len(four_mom[:,0])    
        

#------------------------------------------------------------------------------------------------------------------------------------

#apply cuts and coordinate transformation

main_coords = np.zeros((N, maximum, 3)) #array of all coordinates for all particles for N events
p_stat = 4 * np.ones((N, maximum)) #array of statuses of all particles for N events
main_pid = np.zeros((N, maximum)) #array of pids of all particles for N events
main_daughter1 = np.zeros((N, maximum)) #array of daughter 1s for all particles of N events
main_daughter2 = np.zeros((N, maximum)) #array of daughter 2s for all particles of N events

#do cuts and coordinate transformation

for i in range(N):    
    temp0, temp1, temp2, temp3, temp4, nan1, nan2 = apply_cuts(data, particle_eta_cut, particle_pt_cut, i)
    main_coords[i,0:len(temp0),:] = coord_transform(temp0)
    
    p_stat[i,0:len(temp0)] = temp1
    
    main_pid[i,0:len(temp0)] = temp2
    
    main_daughter1[i,0:len(temp0)] = temp3
    
    main_daughter2[i,0:len(temp0)] = temp4

#remove events that had all particles removed from cuts
zero_array = []

for i in range(N):
    if np.all(main_coords[i] == np.zeros((maximum, 3))):
        zero_array.append(i)
        print(i)


main_coords = np.delete(main_coords, zero_array, axis=0)
p_stat = np.delete(p_stat, zero_array, axis=0)
main_pid = np.delete(main_pid, zero_array, axis = 0)
main_daughter1 = np.delete(main_daughter1, zero_array, axis = 0)
main_daughter2 = np.delete(main_daughter2, zero_array, axis = 0)
print(f'Coordinate transformations and kinematic cuts applied')
N = len(main_coords)
#-------------------------------------------------------------------------------------

#pt and eta,phi thats going to be used for the EMD calculation 
SEMD_data =  main_coords[:,:,0:3]
#construct individual arrays for pT and (eta, phi) for each class of particles for all N events

hadronization_SEMD_data = np.zeros((N,maximum,3))
hardprocess_SEMD_data = np.zeros((N,maximum,3))
showered_SEMD_data = np.zeros((N,maximum,3))


for i in range(N):
    temp_fs = len(np.where((main_daughter1[i] == -1) & (main_daughter2[i] == -1))[0])
    hadronization_SEMD_data[i,0:temp_fs] = SEMD_data[i,np.where((main_daughter1[i] == -1) & (main_daughter2[i] == -1))[0]]
    
    temp_sp = len(np.where(p_stat[i] == 23)[0])
    hardprocess_SEMD_data[i,0:temp_sp] = SEMD_data[i,np.where(p_stat[i] == 23)[0]]
    
    temp_sh = len(np.where((p_stat[i] > 70) & (p_stat[i] < 80))[0])
    showered_SEMD_data[i,0:temp_sh] = SEMD_data[i,np.where((p_stat[i] > 70) & (p_stat[i] < 80))[0]]
if args.hardprocess: 
    zero_hp_array = []
    for i in range(len(hardprocess_SEMD_data)):
        if np.all(hardprocess_SEMD_data[i] ==np.zeros((maximum,3))):
            zero_hp_array.append(i)
            print(i)
    hardprocess_SEMD_data = np.delete(hardprocess_SEMD_data, zero_hp_array, axis = 0)

if args.showered: 
    zero_sh_array = []
    for i in range(len(showered_SEMD_data)):
        if np.all(showered_SEMD_data[i] ==np.zeros((maximum,3))):
            zero_sh_array.append(i)
            print(i)
    showered_SEMD_data = np.delete(showered_SEMD_data, zero_sh_array, axis = 0)

if args.hadronization: 
    zero_had_array = []
    for i in range(len(hadronization_SEMD_data)):
        if np.all(hadronization_SEMD_data[i] ==np.zeros((maximum,3))):
            zero_had_array.append(i)
            print(i)
    hadronization_SEMD_data = np.delete(hadronization_SEMD_data, zero_had_array, axis = 0)
#remove excessive zeroes from pT, eta, phi arrays

max1 = maximum
max2 = maximum
max3 = maximum


for i in range(len(hardprocess_SEMD_data)):
    B = np.count_nonzero(hardprocess_SEMD_data[i,:,0]== 0)
    if B < max2:
        max2 = B

for i in range(len(hadronization_SEMD_data)):
    A = np.count_nonzero(hadronization_SEMD_data[i,:,0]== 0) 
    if A < max1:
        max1 = A

for i in range(len(showered_SEMD_data)):
    C = np.count_nonzero(showered_SEMD_data[i,:,0] == 0)
    if C < max3:
        max3 = C

        
hadronization_SEMD_data = hadronization_SEMD_data[:,0:maximum - max1]
hardprocess_SEMD_data = hardprocess_SEMD_data[:,0:maximum - max2]
showered_SEMD_data = showered_SEMD_data[:,0:maximum - max3]
'''
# Removing bug event 6329
hp_arr = np.delete(hardprocess_SEMD_data, 6329, axis=0)
hardprocess_SEMD_data = hp_arr
s_arr = np.delete(showered_SEMD_data, 6329, axis=0)
showered_SEMD_data = s_arr
had_arr = np.delete(hadronization_SEMD_data, 6329, axis = 0)
hadronization_SEMD_data = had_arr
'''
print(f'Splitting into hard process, showered, hadronization completed')


#----------------------------------------------------------------------------------------------------------------------------------

# Normalizing events for BALANCED OT (comment out for unbalanced, add omega_max to emd calculation)
hard_pts = hardprocess_SEMD_data[:,:,0]
max_pts = hard_pts.sum(axis=1, keepdims=True)
max_pts = np.where(max_pts == 0, 1, max_pts)
hardprocess_SEMD_data[:,:,0] = hard_pts / max_pts
#print("normalized pts: ",hard_specter_data[:,:,0])

sh_pts = showered_SEMD_data[:,:,0]
sh_max_pts = sh_pts.sum(axis=1, keepdims=True)
sh_max_pts = np.where(sh_max_pts == 0, 1, sh_max_pts)
showered_SEMD_data[:,:,0] = sh_pts / sh_max_pts

had_pts = hadronization_SEMD_data[:,:,0]
had_max_pts = had_pts.sum(axis=1, keepdims=True)
had_max_pts = np.where(had_max_pts == 0, 1, had_max_pts)
hadronization_SEMD_data[:,:,0] = had_pts / had_max_pts

#----------------------------------------------------------------------------------------------------------------------------------
#read hepmc file of mc data and store event weights
#xsec = np.array([])

#with pyhepmc.open(args.weight_filepath) as f:
 #   for i in f:
 #       xsec = np.append(xsec, i.weights[0])

#print(f'There are {len(np.where(xsec < 0)[0])} negative weighted events out of {len(xsec)} events')

#weight = 1.455 * 10**4
#event_weight = xsec * weight

#make array of negative weighted events
#neg_events = []

#for i in range(len(event_weight)):
 #   if event_weight[i] < 0:
  #      neg_events.append(i)

if ".hepmc" in args.weight_filepath:
    #read hepmc file of mc data and store event weights
    xsec = np.array([])
    
    with pyhepmc.open(args.weight_filepath) as f:
        for i in f:
            xsec = np.append(xsec, i.weights[0])

        print(f'There are {len(np.where(xsec < 0)[0])} negative weighted events out of {len(xsec)} events')

    weight = 1.455 * 10**4
    event_weight = xsec * weight
elif ".npy" in args.weight_filepath:
    event_weight = np.load(args.weight_filepath)*1.455 * 10**4
else:
    raise ValueError(f"--weight_filepath must be a .hepmc or .npy file, got {args.weight_filepath}")
event_weight = event_weight[:N]
neg_events = np.where(event_weight < 0)[0]
mask = np.ones(N, dtype=bool)
if args.hardprocess:
    mask[zero_hp_array] = False
if args.showered:
    mask[zero_sh_array] = False
if args.hadronization:
    mask[zero_had_array] = False

# negative events that were themselves removed by the cuts no longer exist in the compacted arrays;
# drop them first, then map the surviving original indices onto the compacted indices
neg_events = neg_events[mask[neg_events]]
neg_events = np.where(mask)[0].searchsorted(neg_events)
#-------------------------------------------------------------------------------------------------------------------------------------
def calc_emds(data, neg_events, batch_size):
    neg_idx = np.array(neg_events)
    all_idx = np.arange(data.shape[0])

    # Generate all i,j index pairs (neg x all)
    i_pairs, j_pairs = np.meshgrid(neg_idx, all_idx, indexing='ij')
    i_pairs = i_pairs.flatten()
    j_pairs = j_pairs.flatten()

    pairwise_emds = np.zeros((len(neg_idx), len(all_idx)))

    start_time = time.time()

    for n in range(0, len(i_pairs), batch_size):
        if (n // batch_size) % 100 == 0:
            print(f"Computing Batch {n}-{n+batch_size} of {len(i_pairs)} pairs. Elapsed Time: {time.time() - start_time :.3f} seconds")

        i_batch = i_pairs[n : n + batch_size]
        j_batch = j_pairs[n : n + batch_size]
#        print((data[i_batch]).shape)
        emds = specter.spectralEMD(data[i_batch], data[j_batch],omega_max = 4.2) 
        row_idx = np.searchsorted(neg_idx, i_batch)
        col_idx = j_batch
        pairwise_emds[row_idx, col_idx] = emds

    print(f"Done! Total Elapsed Time: {time.time() - start_time :.3f} seconds")
    return pairwise_emds


#distance from all events to negative event (hard process)
if args.hardprocess:
    print(f'Starting hard process distance matrix calculation...')
    hardprocess_dist2neg = calc_emds(hardprocess_SEMD_data, neg_events,100000)
    #for i in range(len(neg_events)):
    #   hardprocess_close2neg[i] = np.argsort(hardprocess_dist2neg[i])
    
    #hardprocess_close2neg = np.delete(hardprocess_close2neg, 0, axis=1)


    np.save('CPU_SPECTER_hardprocess_dist2neg.npy', hardprocess_dist2neg)
    print(f'Hard process distance matrix computed and saved')


#distance from all events to negative event (showered)
if args.showered:
    print(f'Starting showered distance matrix calculation...')
    showered_dist2neg = calc_emds(showered_SEMD_data, neg_events, 10000)

    np.save('SPECTER_showered_dist2neg_GPU_w_4_2.npy', showered_dist2neg)
    print(f'Showered distance matrix computed and saved')


#distance from all events to negative event (hadronization)
if args.hadronization:
    print(f'Starting hadronization distance matrix calculation...')
    hadronization_dist2neg = calc_emds(hadronization_SEMD_data,neg_events,1000)

    np.save('SPECTER_hadronization_dist2neg.npy', hadronization_dist2neg)
    print(f'Hadronization distance matrix computed and saved')
