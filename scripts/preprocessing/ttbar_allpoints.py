import numpy as np
import uproot
import os
import fastjet as fj
import argparse


#check number of cores and compile specter
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nevents", type = int, required = True, help = "Number of events to make points for")
args = parser.parse_args()

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
     
#    print(removal_idx)
    #print(num, "particles removed out of", num_particles)

    
    return four_momenta, status, pid, daughter1, daughter2, mass, f"{num} particles removed out of {num_particles}"


#function to transform into (pT, eta, phi) coordinates

def coord_transform(data):
    
    num_particles = len(data[:,1])
    
    new_coords = np.zeros((num_particles, 3))
    
    new_coords[:,0] = np.sqrt(data[:,1]**2 + data[:,2]**2)
    new_coords[:,1] = np.arctanh(data[:,3]/np.sqrt(data[:,1]**2 + data[:,2]**2 + data[:,3]**2))
    new_coords[:,2] = np.arctan2(data[:,2],data[:,1]) 
            
    
    return new_coords



file = uproot.open('ttbar_NLO_100k.root')
#file = uproot.open('/users/rrjain/ppzjj_samples/0_zpt_cut.root')
tree = file["Events"]



#define cut parameters

particle_eta_cut = 4.9
particle_pt_cut = 0.1

N = args.nevents #number of events

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
        
print("maximum n particles in event ", maximum)


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

print("cuts applied and coordinates ransformed")

for i in range(N):
    if np.all(main_coords[i] == np.zeros((maximum, 3))):
        zero_array.append(i)
        print(i)

main_coords = np.delete(main_coords, zero_array, axis=0)
p_stat = np.delete(p_stat, zero_array, axis=0)
main_pid = np.delete(main_pid, zero_array, axis = 0)
main_daughter1 = np.delete(main_daughter1, zero_array, axis = 0)
main_daughter2 = np.delete(main_daughter2, zero_array, axis = 0)



#p and eta,phi thats going to be used for the EMD calculation - NOTE THIS IS DIFFERENT THAN THE EVENT WEIGHT
EMD_pt = main_coords[:,:,0]
EMD_etaphi = main_coords[:,:,1:3]

#construct individual arrays for pT and (eta, phi) for each class of particles for all N events

hadronization_EMD_pt = np.zeros((N,maximum))
hardprocess_EMD_pt = np.zeros((N,maximum))
showered_EMD_pt = np.zeros((N,maximum))

hadronization_EMD_etaphi = np.zeros((N,maximum,2))
hardprocess_EMD_etaphi = np.zeros((N,maximum,2))
showered_EMD_etaphi = np.zeros((N,maximum, 2))

for i in range(N):
    temp_fs = len(np.where((main_daughter1[i] == -1) & (main_daughter2[i] == -1))[0])
    hadronization_EMD_pt[i,0:temp_fs] = EMD_pt[i,np.where((main_daughter1[i] == -1) & (main_daughter2[i] == -1))[0]]
    hadronization_EMD_etaphi[i,0:temp_fs] = EMD_etaphi[i,np.where((main_daughter1[i] == -1) & (main_daughter2[i] == -1))[0]]
    
    temp_sp = len(np.where(p_stat[i] == 23)[0])
    hardprocess_EMD_pt[i,0:temp_sp] = EMD_pt[i,np.where(p_stat[i] == 23)[0]]
    hardprocess_EMD_etaphi[i,0:temp_sp] = EMD_etaphi[i,np.where(p_stat[i] == 23)[0]]

    
    temp_sh = len(np.where((p_stat[i] > 70) & (p_stat[i] < 80))[0])
    showered_EMD_pt[i,0:temp_sh] = EMD_pt[i,np.where((p_stat[i] > 70) & (p_stat[i] < 80))[0]]
    showered_EMD_etaphi[i,0:temp_sh] = EMD_etaphi[i,np.where((p_stat[i] > 70) & (p_stat[i] < 80))[0]]



#remove excessive zeroes from pT, eta, phi arrays

max1 = maximum
max2 = maximum
max3 = maximum

for i in range(N):
    A = np.count_nonzero(hadronization_EMD_pt[i] == 0)
    if A < max1:
        max1 = A
    
    B = np.count_nonzero(hardprocess_EMD_pt[i] == 0)
    if B < max2:
        max2 = B
        
    C = np.count_nonzero(showered_EMD_pt[i] == 0)
    if C < max3:
        max3 = C

        
hadronization_EMD_pt = hadronization_EMD_pt[:,0:maximum - max1]
hardprocess_EMD_pt = hardprocess_EMD_pt[:,0:maximum - max2]
showered_EMD_pt = showered_EMD_pt[:,0:maximum - max3]

hadronization_EMD_etaphi = hadronization_EMD_etaphi[:,0:maximum - max1]
hardprocess_EMD_etaphi = hardprocess_EMD_etaphi[:,0:maximum - max2]
showered_EMD_etaphi = showered_EMD_etaphi[:,0:maximum - max3]

print(maximum - max1)
print(maximum - max2)
print(maximum - max3)

hardprocess_points = np.concatenate((np.expand_dims(hardprocess_EMD_pt, axis = -1), hardprocess_EMD_etaphi), axis = -1)
showered_points = np.concatenate((np.expand_dims(showered_EMD_pt, axis = -1), showered_EMD_etaphi), axis = -1)
hadronization_points = np.concatenate((np.expand_dims(hadronization_EMD_pt, axis = -1), hadronization_EMD_etaphi), axis = -1)

np.save('hardprocess_ttbar_points10k.npy', hardprocess_points)
np.save('showered_ttbar_points10k.npy', showered_points)
np.save('hadronization_ttbar_points10k.npy', hadronization_points)

