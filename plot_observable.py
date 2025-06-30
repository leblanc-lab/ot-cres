#Import useful stuff
import ROOT
import uproot
import numpy as np
import matplotlib.pyplot as plt
import fastjet as fj
import awkward as ak
import vector
import os
import pyhepmc
import csv
import random
from array import array
import argparse
import sys

#check number of cores
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

#import arguments
parser = argparse.ArgumentParser()

parser.add_argument("--hard_distmatrix", type=str, required = True, help = "Path to hard process distance matrix. Must be a csv file")
parser.add_argument("--showered_distmatrix", type = str, required = True, help = "Path to showered distance matrix. Must be a csv file")
parser.add_argument("--hadron_distmatrix", type = str, required = True, help = "Path to hadronization distance matrix. Must be a csv file")
parser.add_argument("--max_radius", type = float, required = True, help = "Max cell radius during reweighting. Must be a positive number")
parser.add_argument("--event_filepath", type = str, required = True, help = "Path to event root file")
parser.add_argument("--small_weight_filepath", type=str, required=True, help="Path to the hepmc file of small data weights")
parser.add_argument("--big_weight_filepath", type = str, required=True, help="Path to .npy file of big data weights")
parser.add_argument("--big_obs_filepath", type = str, required=True, help="Path to .npy file of big data observable")
parser.add_argument("--big_obs_filepath2", type = str, required = False, help = "Path to .npy file for 2nd big data observable (if needed)")
parser.add_argument("--N", type = int, required = True, help = "Number of events to look at")
parser.add_argument("--obs", type = str, required = True, help = "What observable do you want to visualize?")
parser.add_argument("--numbins", type = int, required = True, help = "Number of bins in histogram")
parser.add_argument("--logscale", action='store_true', help = "Is y axis of plot log scaled or no?")
parser.add_argument("--logbins", action='store_true', help = "Are bin edges log scaled? Only use if all bins are positive")
parser.add_argument("--minbins", type = int, required = False, help = "If logbins = true, this specifies the minimum size a bin can be")
parser.add_argument("--jet_obs", action='store_true', help = "Is the observable a jet observable?")
parser.add_argument("--jet_pt_cutoff", type = int, required = False, help = "Required if jet_obs = true, this specifies the lower pt cutoff for jet clustering")

args = parser.parse_args()
N = args.N
obs = args.obs

#print all supported observables
print(f'\n \n \n')
print(f'Currently supported observables: \nht: ht of event \nzpt: Z boson pt \nzy: Z boson rapidity \nzphi: Z boson azimuthal angle\nl_jet_pt: Leading jet pt \nsl_jet_pt: Subleading jet pt')
print(f'l_jet_eta: Leading jet rapidity \nsl_jet_eta: Subleading jet rapidity \nl_jet_phi: Leading jet azimuthal angle \nsl_jet_phi: Subleading jet azimuthal angle')
print(f'jet_dist: Distance between leading and subleading jets \nz_ljet_dist: Distance between Z and leading jet \nz_sljet_dist: distance between Z and subleading jet')
print(f'ljetpt_over_ht: Leading jet pt fraction \nsljetpt_over_ht: Subleading jet pt fraction \nzpt_over_ht: Z boson pt fraction \nljetpt_over_sljetpt: Leading jet pt / Subleading jet pt')
print(f'z_ljet_delta_phi: Δphi between Z and leading jet \nz_sljet_delta_phi: Δphi between Z and subleading jet \nljet_sljet_delta_phi: Δphi between leading and subleading jets')
print(f'z_ljet_delta_eta: Δeta between Z and leading jet \nz_sljet_delta_eta: Δeta between Z and subleading jet \nljet_sljet_delta_eta: Δeta between leading and subleading jets')
print(f'n_jets: Number of jets above pt threshold')
print(f'\n \n \n')

#print what inputs are needed for multi-input observables
if obs in ('jet_dist', 'ljet_sljet_delta_phi', 'ljet_sljet_delta_eta', 'ljetpt_over_sljetpt'):
    print(F'WARNING: THIS OBSERVABLE NEEDS THE FOLLOWING INPUTS: \nbig_obs_filepath: leading jet coords filepath \nbig_obs_filepath2: subleading jet coords filepath')
if obs in ('z_ljet_dist', 'z_ljet_delta_phi', 'z_ljet_delta_eta'):
    print(F'WARNING: THIS OBSERVABLE NEEDS THE FOLLOWING INPUTS: \nbig_obs_filepath: leading jet coords filepath \nbig_obs_filepath2: Z coords filepath')
if obs in ('z_sljet_dist', 'z_sljet_delta_phi', 'z_sljet_delta_eta'):
    print(F'WARNING: THIS OBSERVABLE NEEDS THE FOLLOWING INPUTS: \nbig_obs_filepath: subleading jet coords filepath \nbig_obs_filepath2: Z coords filepath')
if obs == 'ljetpt_over_ht':
    print(F'WARNING: THIS OBSERVABLE NEEDS THE FOLLOWING INPUTS: \nbig_obs_filepath: leading jet coords filepath \nbig_obs_filepath2: ht filepath')
if obs == 'sljetpt_over_ht':
    print(F'WARNING: THIS OBSERVABLE NEEDS THE FOLLOWING INPUTS: \nbig_obs_filepath: subleading jet coords filepath \nbig_obs_filepath2: ht filepath')
if obs == 'zpt_over_ht':
    print(F'WARNING: THIS OBSERVABLE NEEDS THE FOLLOWING INPUTS: \nbig_obs_filepath: Z coords filepath \nbig_obs_filepath2: ht filepath')
print(f'\n')
#----------------------------------------------------------------------------------------------------------------------------------------------
#define important functions

#function to apply cuts to data

def apply_cuts(data, ycut, ptcut, event_num):
    
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
        
        elif abs(np.arctanh(four_momenta[i,3]/four_momenta[i,0])) > ycut:
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
    new_coords[:,1] = 1/2*np.log((data[:,0] + data[:,3])/(data[:,0] - data[:,3]))
    new_coords[:,2] = np.arctan2(data[:,2],data[:,1]) 
            
    
    return new_coords

#---------------------------------------------------------------------------------------------------------------------------
#import data
event_filepath = args.event_filepath
file = uproot.open(event_filepath)
tree = file["Events"]


#define cut parameters

particle_y_cut = 4.9
particle_pt_cut = 0.1

N = args.N  #number of events to analyze

#create data of 4 momenta
data = tree.arrays(["Particle_energy", "Particle_px","Particle_py", "Particle_pz", "Particle_status", "Particle_pid", 
                    "Particle_d1", "Particle_d2", "Particle_mass"])
data = data[0:N]


#which event has the most particles? This will set the size of higher dimensional space
maximum = 0

for i in range(N):
    four_mom = apply_cuts(data, particle_y_cut, particle_pt_cut, i)[0]
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
    temp0, temp1, temp2, temp3, temp4, nan1, nan2 = apply_cuts(data, particle_y_cut, particle_pt_cut, i)
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

#-------------------------------------------------------------------------------------

#pt and eta,phi thats going to be used for the EMD calculation 
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

print(f'Splitting into hard process, showered, hadronization completed')

#----------------------------------------------------------------------------------------------------------------------------------
#read hepmc file of mc data and store event weights
xsec = np.array([])

with pyhepmc.open(args.small_weight_filepath) as f:
    for i in f:
        xsec = np.append(xsec, i.weights[0])

print(f'There are {len(np.where(xsec < 0)[0])} negative weighted events out of {len(xsec)} events')

weight = 1.455 * 10**4
event_weight = xsec * weight

#make array of negative weighted events
neg_events = []

for i in range(len(event_weight)):
    if event_weight[i] < 0:
        neg_events.append(i)
        
#-------------------------------------------------------------------------------------------------------------------------------------
#load presaved distance matrices
hardprocess_dist2neg = np.loadtxt(args.hard_distmatrix, delimiter=',')
showered_dist2neg = np.loadtxt(args.showered_distmatrix, delimiter=',')
hadronization_dist2neg = np.loadtxt(args.hadron_distmatrix, delimiter=',')

#compute proximity matrices
hardprocess_close2neg = np.zeros((len(neg_events), args.N))

for i in range(len(neg_events)):
    hardprocess_close2neg[i] = np.argsort(hardprocess_dist2neg[i])
    
hardprocess_close2neg = np.delete(hardprocess_close2neg, 0, axis=1)



showered_close2neg = np.zeros((len(neg_events), args.N))

for i in range(len(neg_events)):
    showered_close2neg[i] = np.argsort(showered_dist2neg[i])
    
showered_close2neg = np.delete(showered_close2neg, 0, axis=1)



hadronization_close2neg = np.zeros((len(neg_events), args.N))

for i in range(len(neg_events)):
    hadronization_close2neg[i] = np.argsort(hadronization_dist2neg[i])
    
hadronization_close2neg = np.delete(hadronization_close2neg, 0, axis=1)

print('Distance and proximity matrices constructed')
#--------------------------------------------------------------------------------------------------------
max_radius = args.max_radius

#reshuffle distance and proximity matrices by most isolated negative events

hardprocess_nearest_neighbor = []
showered_nearest_neighbor = []
hadronization_nearest_neighbor = []


for i in range(len(neg_events)):
    hardprocess_nearest_neighbor.append(np.min([j for j in hardprocess_dist2neg[i] if j > 0]))
    showered_nearest_neighbor.append(np.min([k for k in showered_dist2neg[i] if k > 0]))
    hadronization_nearest_neighbor.append(np.min([l for l in hadronization_dist2neg[i] if l > 0]))


hardprocess_nearest_neighbor = np.argsort(-np.array(hardprocess_nearest_neighbor))
showered_nearest_neighbor = np.argsort(-np.array(showered_nearest_neighbor))
hadronization_nearest_neighbor = np.argsort(-np.array(hadronization_nearest_neighbor))


hardprocess_close2neg_reshuffled = hardprocess_close2neg[hardprocess_nearest_neighbor]
hardprocess_dist2neg_reshuffled = hardprocess_dist2neg[hardprocess_nearest_neighbor]
hardprocess_dist2neg_reshuffled = np.sort(hardprocess_dist2neg_reshuffled, axis=1)
hardprocess_dist2neg_reshuffled = np.delete(hardprocess_dist2neg_reshuffled, 0, axis=1)

showered_close2neg_reshuffled = showered_close2neg[showered_nearest_neighbor]
showered_dist2neg_reshuffled = showered_dist2neg[showered_nearest_neighbor]
showered_dist2neg_reshuffled = np.sort(showered_dist2neg_reshuffled, axis=1)
showered_dist2neg_reshuffled = np.delete(showered_dist2neg_reshuffled, 0, axis=1)

hadronization_close2neg_reshuffled = hadronization_close2neg[hadronization_nearest_neighbor]
hadronization_dist2neg_reshuffled = hadronization_dist2neg[hadronization_nearest_neighbor]
hadronization_dist2neg_reshuffled = np.sort(hadronization_dist2neg_reshuffled, axis=1)
hadronization_dist2neg_reshuffled = np.delete(hadronization_dist2neg_reshuffled, 0, axis=1)

print('Initial reshuffling completed')
#reweighting for hard process events

hardprocess_event_weight = np.copy(np.array(event_weight))
hardprocess_cell_radius = []

for i in range(N):     #search through all events
    if hardprocess_event_weight[i] < 0:     #if the i-th event has negative weight
        hardprocess_cell_weight = hardprocess_event_weight[i]     #cell weight = seed weight -> other event weights will be added to this
        hardprocess_abs_cell_weight = abs(hardprocess_event_weight[i]) #sum of absolute value of weights (needed for jeppe's technique)
        hardprocess_events_in_cell = [i]     #index of events within cell

        for j in range(N):      #search through all events
            if hardprocess_cell_weight <= 0 and hardprocess_dist2neg_reshuffled[np.where(np.array(neg_events) == i)[0],j] < max_radius:#if the cell weight is negative and within max radius
                hardprocess_cell_weight = np.append(hardprocess_cell_weight,
                                                    hardprocess_event_weight[int(hardprocess_close2neg_reshuffled[np.where(np.array(neg_events) == i )[0],j])])      

                hardprocess_abs_cell_weight += abs(hardprocess_cell_weight[1]) #add the abs of weight of that event to cell weight
                hardprocess_cell_weight = np.sum(hardprocess_cell_weight)     #add the weight of that event to the cell weight
                hardprocess_events_in_cell = np.append(hardprocess_events_in_cell, hardprocess_close2neg_reshuffled[np.where(np.array(neg_events) == i )[0],j])     #add that event to the cell
            else:
                break


        if hardprocess_cell_weight > 0:
            hardprocess_cell_radius = np.append(hardprocess_cell_radius, hardprocess_dist2neg_reshuffled[np.where(np.array(neg_events) == i)[0],j - 1])
    

    
            hardprocess_event_weight[hardprocess_events_in_cell.astype(int)] = hardprocess_cell_weight / hardprocess_abs_cell_weight * abs(hardprocess_event_weight[hardprocess_events_in_cell.astype(int)]) #jeppe reweighting

print(f'\n')
print('Hard process reweighting completed')
print(f'\n')
#reweighting for showered events

showered_event_weight = np.copy(np.array(event_weight))
showered_cell_radius = []

for i in range(N):     #search through all events
    if showered_event_weight[i] < 0:     #if the i-th event has negative weight
        showered_cell_weight = showered_event_weight[i]     #cell weight = seed weight -> other event weights will be added to this
        showered_abs_cell_weight = abs(showered_event_weight[i]) #sum of absolute value of weights (needed for jeppe's technique)
        showered_events_in_cell = [i]     #index of events within cell

        for j in range(N):      #search through all events
            if showered_cell_weight <= 0 and showered_dist2neg_reshuffled[np.where(np.array(neg_events) == i)[0],j] < max_radius:#if the cell weight is negative and within max radius
                showered_cell_weight = np.append(showered_cell_weight,
                                                    showered_event_weight[int(showered_close2neg_reshuffled[np.where(np.array(neg_events) == i )[0],j])])      

                showered_abs_cell_weight += abs(showered_cell_weight[1]) #add the abs of weight of that event to cell weight
                showered_cell_weight = np.sum(showered_cell_weight)     #add the weight of that event to the cell weight
                showered_events_in_cell = np.append(showered_events_in_cell, showered_close2neg_reshuffled[np.where(np.array(neg_events) == i )[0],j])     #add that event to the cell
                #print(hardprocess_dist2neg_reshuffled[np.where(np.array(neg_events) == i)[0],j],i,j)
            else:
                break

        if showered_cell_weight > 0:
            showered_cell_radius = np.append(showered_cell_radius, showered_dist2neg_reshuffled[np.where(np.array(neg_events) == i)[0],j - 1])

    
            showered_event_weight[showered_events_in_cell.astype(int)] = showered_cell_weight / showered_abs_cell_weight * abs(showered_event_weight[showered_events_in_cell.astype(int)]) #jeppe reweighting


print(f'\n')
print('Showered reweighting completed')
print(f'\n')
#reweighting for hadronized events

hadronization_event_weight = np.copy(np.array(event_weight))
hadronization_cell_radius = []

for i in range(N):     #search through all events
    if hadronization_event_weight[i] < 0:     #if the i-th event has negative weight
        hadronization_cell_weight = hadronization_event_weight[i]     #cell weight = seed weight -> other event weights will be added to this
        hadronization_abs_cell_weight = abs(hadronization_event_weight[i]) #sum of absolute value of weights (needed for jeppe's technique)
        hadronization_events_in_cell = [i]     #index of events within cell

        for j in range(N):      #search through all events
            if hadronization_cell_weight <= 0 and hadronization_dist2neg_reshuffled[np.where(np.array(neg_events) == i)[0],j] < max_radius:#if the cell weight is negative and within max radius
                hadronization_cell_weight = np.append(hadronization_cell_weight,
                                                    hadronization_event_weight[int(hadronization_close2neg_reshuffled[np.where(np.array(neg_events) == i )[0],j])])      

                hadronization_abs_cell_weight += abs(hadronization_cell_weight[1]) #add the abs of weight of that event to cell weight
                hadronization_cell_weight = np.sum(hadronization_cell_weight)     #add the weight of that event to the cell weight
                hadronization_events_in_cell = np.append(hadronization_events_in_cell, hadronization_close2neg_reshuffled[np.where(np.array(neg_events) == i )[0],j])     #add that event to the cell
            else:
                break


        if hadronization_cell_weight > 0:
            hadronization_cell_radius = np.append(hadronization_cell_radius, hadronization_dist2neg_reshuffled[np.where(np.array(neg_events) == i)[0],j - 1])


    
            hadronization_event_weight[hadronization_events_in_cell.astype(int)] = hadronization_cell_weight / hadronization_abs_cell_weight * abs(hadronization_event_weight[hadronization_events_in_cell.astype(int)]) #jeppe reweighting

print(f'\n')
print('Hadronization reweighting completed')
print(f'\n')
#un-reshuffle the former negative weights after reweighting. This is so that the event numbers in the weight arrays correspond to actual event numbers

backtransform_hardprocess_idx = np.empty_like(hardprocess_nearest_neighbor)
backtransform_hardprocess_idx[hardprocess_nearest_neighbor] = np.arange(len(hardprocess_nearest_neighbor))

backtransform_showered_idx = np.empty_like(showered_nearest_neighbor)
backtransform_showered_idx[showered_nearest_neighbor] = np.arange(len(showered_nearest_neighbor))

backtransform_hadronization_idx = np.empty_like(hadronization_nearest_neighbor)
backtransform_hadronization_idx[hadronization_nearest_neighbor] = np.arange(len(hadronization_nearest_neighbor))




hardprocess_event_weight[neg_events] = hardprocess_event_weight[neg_events][backtransform_hardprocess_idx]
showered_event_weight[neg_events] = showered_event_weight[neg_events][backtransform_showered_idx]
hadronization_event_weight[neg_events] = hadronization_event_weight[neg_events][backtransform_hadronization_idx]

print('All reshuffling completed')
#-----------------------------------------------------------------------------------------------------------------------
#load big event weight
big_event_weight = np.load(args.big_weight_filepath)

rescale = np.sum(big_event_weight) / np.sum(xsec)
event_weight *= rescale

big_event_weight *= weight

if abs(1 - np.sum(big_event_weight) / np.sum(event_weight)) < 10**-5:
    print('Rescaling done successfully')
else:
    print("Reweighting Failed. Exiting")
    sys.exit(1)

#----------------------------------------------------------------------------------------------------------------------
#Do jet clustering

if args.jet_obs == True:
    R = 0.4
    akt_jetdef = fj.JetDefinition(fj.antikt_algorithm, R)

#extract mass from original dataset and separate out hadronization particle masses
    p_mass = np.zeros((N, maximum))
    hadronization_p_mass = np.zeros((N, maximum))
    
    for i in range(N):
        p_mass[i,0:len(apply_cuts(data, particle_y_cut, particle_pt_cut, i)[5])] = apply_cuts(data, particle_y_cut, particle_pt_cut, i)[5]
    
    max_temp_fs = 0 
    
    for i in range(N):
        temp_fs = len(np.where((main_daughter1[i] == -1) & (main_daughter2[i] == -1))[0])
        hadronization_p_mass[i, 0:temp_fs] = p_mass[i,np.where((main_daughter1[i] == -1) & (main_daughter2[i] == -1))[0]]
        if temp_fs > max_temp_fs:
            max_temp_fs = temp_fs
    
    hadronization_p_mass = hadronization_p_mass[:,0:max_temp_fs]


    #create awkward arrays with (pt, eta, phi, mass) to feed into fastjet for clustering
    
    #do the jet clustering using akt algorithm
    N = args.N
    
    jet_pt_cutoff = args.jet_pt_cutoff
    
    vector.register_awkward()
    
    l_jet = ak.Array([{"px": None, "py": None, "pz": None, "E": None}], with_name="Momentum4D")[0:0]
    sl_jet = ak.Array([{"px": None, "py": None, "pz": None, "E": None}], with_name="Momentum4D")[0:0]
    
    n_jets = []
    
    
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
        akt_jets = fj.sorted_by_pt(akt_cluster.inclusive_jets())
    
        l_jet = ak.concatenate([l_jet, [akt_jets[-1]]])
        sl_jet = ak.concatenate([sl_jet, [akt_jets[-2]]])
    
    
        #create n_jets observable
        n_jet_counter = 0
        for k in range(len(akt_jets)):
            if np.sqrt(akt_jets[k]['px']**2 + akt_jets[k]['py']**2) > jet_pt_cutoff:
    
                n_jet_counter += 1
    
        
        n_jets = np.append(n_jets, n_jet_counter)
    
        if j % 1000 == 0:
            print(f'Completed Events: {j} / {N}')

    print('Jet clustering complete!')

else:
    print('No jet clustering done')

###################################################################################################################################################
######################################################### define all observables ##################################################################
###################################################################################################################################################
obs = args.obs
observable = np.zeros(N)
big_filepath = args.big_obs_filepath
big_observable = np.load(big_filepath)

#ht
if obs == 'ht':
    observable = np.zeros(N)
    for i in range(N):
        observable[i] = np.sum(hadronization_EMD_pt[i])

    title = r'h_{T}'

#zpt
if obs == 'zpt':
    observable = np.zeros((N,1))
    for i in range(N):
        observable[i] = np.sqrt((data[i]['Particle_px'][np.where(data[i]['Particle_status'] == 22)[0]]**2+data[i]['Particle_py'][np.where(data[i]['Particle_status'] == 22)[0]]**2))
    observable = observable[:,0]

    big_observable = big_observable[:,0]

    title = r'Z p_{T}'

#zy
if obs == 'zy':
    observable = np.zeros((N,1)) 
    for i in range(N):
        observable[i] = np.arctanh((data[i]['Particle_pz'][np.where(data[i]['Particle_status'] == 22)[0]]/data[i]['Particle_energy'][np.where(data[i]['Particle_status'] == 22)[0]])) 
    observable = observable[:,0]

    big_observable = big_observable[:,1]

    title = r'Z \eta'

#zphi
if obs == 'zphi':
    observable = np.zeros((N,1))
    for i in range(N):
        observable[i] = np.arctan2(data[i]['Particle_py'][np.where(data[i]['Particle_status'] == 22)[0]],data[i]['Particle_px'][np.where(data[i]['Particle_status'] == 22)[0]])
    observable = observable[:,0]

    big_observable = big_observable[:,2]

    title = r'Z \phi'


#lead jet pt
if obs == 'l_jet_pt':
    for i in range(N):
        observable[i] = np.sqrt(l_jet[i]['px']**2 + l_jet[i]['py']**2)
    big_observable = big_observable[:,0]

    title = r'Leading Jet p_{T}'

#sublead jet pt
if obs == 'sl_jet_pt':
    for i in range(N):
        observable[i] = np.sqrt(sl_jet[i]['px']**2 + sl_jet[i]['py']**2)
    big_observable = big_observable[:,0]

    title = r'Subleading Jet p_{T}'

#lead jet eta
if obs == 'l_jet_eta':
    for i in range(N):
        observable[i] = np.arctanh(l_jet[i]['pz']/l_jet[i]['E'])
    big_observable = big_observable[:,1]  

    title = r'Leading Jet \eta'

#sublead jet eta
if obs == 'sl_jet_eta':
    for i in range(N):
        observable[i] = np.arctanh(sl_jet[i]['pz']/sl_jet[i]['E'])
    big_observable = big_observable[:,1]    

    title = r'Subleading Jet \eta'

#lead jet phi
if obs == 'l_jet_phi':
    for i in range(N):
        observable[i] = np.arctan2(l_jet[i]['py'],l_jet[i]['px'])
    big_observable = big_observable[:,2]

    title = r'Leading Jet \phi'

#sublead jet phi
if obs == 'sl_jet_phi':
    for i in range(N):
        observable[i] = np.arctan2(sl_jet[i]['py'],sl_jet[i]['px'])
    big_observable = big_observable[:,2]

    title = r'Subleading Jet \phi'

#Distance between leading and subleading jets
if obs == 'jet_dist':
    l_jet_eta = np.zeros(N)
    sl_jet_eta = np.zeros(N)
    l_jet_phi = np.zeros(N)
    sl_jet_phi = np.zeros(N)
    
    for i in range(N):
        l_jet_eta[i] = np.arctanh(l_jet[i]['pz']/l_jet[i]['E'])
        sl_jet_eta[i] = np.arctanh(sl_jet[i]['pz']/sl_jet[i]['E'])
        l_jet_phi[i] = np.arctan2(l_jet[i]['py'],l_jet[i]['px'])
        sl_jet_phi[i] = np.arctan2(sl_jet[i]['py'],sl_jet[i]['px'])
        observable[i] = np.sqrt((l_jet_phi[i] - sl_jet_phi[i])**2 + (l_jet_eta[i] - sl_jet_eta[i])**2)

    big_filepath2 = args.big_obs_filepath2
    big_observable2 = np.load(big_filepath2)
    
    big_ljet_eta = big_observable[:,1]
    big_ljet_phi = big_observable[:,2]
    big_sljet_eta = big_observable2[:,1]
    big_sljet_phi = big_observable2[:,2]

    big_observable = np.sqrt((big_ljet_eta - big_sljet_eta)**2 + (big_ljet_phi - big_sljet_phi)**2)

    title = r'\Delta R Between Leading, Subleading Jets'


#Distance between Z and leading jet
if obs == 'z_ljet_dist':
    l_jet_eta = np.zeros(N)
    z_eta = np.zeros((N,1))
    l_jet_phi = np.zeros(N)
    z_phi = np.zeros((N,1))
    
    for i in range(N):
        l_jet_eta[i] = np.arctanh(l_jet[i]['pz']/l_jet[i]['E'])
        z_eta[i] = np.arctanh((data[i]['Particle_pz'][np.where(data[i]['Particle_status'] == 22)[0]]/data[i]['Particle_energy'][np.where(data[i]['Particle_status'] == 22)[0]]))
        l_jet_phi[i] = np.arctan2(l_jet[i]['py'],l_jet[i]['px'])
        z_phi[i] = np.arctan2(data[i]['Particle_py'][np.where(data[i]['Particle_status'] == 22)[0]],data[i]['Particle_px'][np.where(data[i]['Particle_status'] == 22)[0]])

    z_eta = z_eta[:,0]
    z_phi = z_phi[:,0]
    observable = np.sqrt((l_jet_phi - z_phi)**2 + (l_jet_eta - z_eta)**2)
    
    big_filepath2 = args.big_obs_filepath2
    big_observable2 = np.load(big_filepath2)

    big_ljet_eta = big_observable[:,1]
    big_ljet_phi = big_observable[:,2]
    big_z_eta = big_observable2[:,1]
    big_z_phi = big_observable2[:,2]

    big_observable = np.sqrt((big_ljet_eta - big_z_eta)**2 + (big_ljet_phi - big_z_phi)**2)

    title = r'\Delta R Between Z, Leading Jet'


#Distance between Z and subleading jet
if obs == 'z_sljet_dist':
    sl_jet_eta = np.zeros(N)
    z_eta = np.zeros((N,1))
    sl_jet_phi = np.zeros(N)
    z_phi = np.zeros((N,1))
    
    for i in range(N):
        sl_jet_eta[i] = np.arctanh(sl_jet[i]['pz']/sl_jet[i]['E'])
        z_eta[i] = np.arctanh((data[i]['Particle_pz'][np.where(data[i]['Particle_status'] == 22)[0]]/data[i]['Particle_energy'][np.where(data[i]['Particle_status'] == 22)[0]]))
        sl_jet_phi[i] = np.arctan2(sl_jet[i]['py'],sl_jet[i]['px'])
        z_phi[i] = np.arctan2(data[i]['Particle_py'][np.where(data[i]['Particle_status'] == 22)[0]],data[i]['Particle_px'][np.where(data[i]['Particle_status'] == 22)[0]])

    z_eta = z_eta[:,0]
    z_phi = z_phi[:,0]
    observable = np.sqrt((sl_jet_phi - z_phi)**2 + (sl_jet_eta - z_eta)**2)
    
    big_filepath2 = args.big_obs_filepath2    
    big_observable2 = np.load(big_filepath2)

    big_sljet_eta = big_observable[:,1]
    big_sljet_phi = big_observable[:,2]
    big_z_eta = big_observable2[:,1]
    big_z_phi = big_observable2[:,2]

    big_observable = np.sqrt((big_sljet_eta - big_z_eta)**2 + (big_sljet_phi - big_z_phi)**2)

    title = r'\Delta R Between Z, Subleading Jet'

#leading jet pt / ht
if obs == 'ljetpt_over_ht':
    ljet_pt = np.zeros(N)
    ht = np.zeros(N)

    for i in range(N):
        ht[i] = np.sum(hadronization_EMD_pt[i])
        ljet_pt[i] = np.sqrt(l_jet[i]['px']**2 + l_jet[i]['py']**2)

    observable = ljet_pt / ht

    big_ljet_pt = big_observable[:,0]
    big_filepath2 = args.big_obs_filepath2
    big_observable2 = np.load(big_filepath2)

    big_observable = big_ljet_pt / big_observable2

    title = r'Leading Jet p_{T} / h_{T}'


#subleading jet pt / ht
if obs == 'sljetpt_over_ht':
    sljet_pt = np.zeros(N)
    ht = np.zeros(N)

    for i in range(N):
        ht[i] = np.sum(hadronization_EMD_pt[i])
        sljet_pt[i] = np.sqrt(sl_jet[i]['px']**2 + sl_jet[i]['py']**2)

    observable = sljet_pt / ht

    big_sljet_pt = big_observable[:,0]
    big_filepath2 = args.big_obs_filepath2
    big_observable2 = np.load(big_filepath2)

    big_observable = big_sljet_pt / big_observable2

    title = r'Subleading Jet p_{T} / h_{T}'


#Z pt / ht
if obs == 'zpt_over_ht':
    zpt = np.zeros((N,1))
    ht = np.zeros(N)

    for i in range(N):
        ht[i] = np.sum(hadronization_EMD_pt[i])
        zpt[i] = np.sqrt((data[i]['Particle_px'][np.where(data[i]['Particle_status'] == 22)[0]]**2+data[i]['Particle_py'][np.where(data[i]['Particle_status'] == 22)[0]]**2))

    zpt = zpt[:,0]
    observable = zpt / ht

    big_zpt = big_observable[:,0]
    big_filepath2 = args.big_obs_filepath2
    big_observable2 = np.load(big_filepath2)

    big_observable = big_zpt / big_observable2

    title = r'Z p_{T} / h_{T}'


#leading jet pt over subleading jet pt
if obs == 'ljetpt_over_sljetpt':
    ljet_pt = np.zeros(N)
    sljet_pt = np.zeros(N)

    for i in range(N):
        ljet_pt[i] = np.sqrt(l_jet[i]['px']**2 + l_jet[i]['py']**2)
        sljet_pt[i] = np.sqrt(sl_jet[i]['px']**2 + sl_jet[i]['py']**2)

    observable = ljet_pt / sljet_pt

    big_ljet_pt = big_observable[:,0]
    big_filepath2 = args.big_obs_filepath2
    big_observable2 = np.load(big_filepath2)
    big_observable2 = big_observable2[:,0]

    big_observable = big_ljet_pt / big_observable2

    title = r'Leading Jet p_{T} / Subleading Jet p_{T}'


#delta phi between Z and leading jet 
if obs == 'z_ljet_delta_phi':
    z_phi = np.zeros((N,1))
    l_jet_phi = np.zeros(N)

    for i in range(N):
        l_jet_phi[i] = np.arctan2(l_jet[i]['py'],l_jet[i]['px'])
        z_phi[i] = np.arctan2(data[i]['Particle_py'][np.where(data[i]['Particle_status'] == 22)[0]],data[i]['Particle_px'][np.where(data[i]['Particle_status'] == 22)[0]])

    z_phi = z_phi[:,0]

    observable = abs(zphi - l_jet_phi)

    big_ljet_phi = big_observable[:,2]
    big_filepath2 = args.big_obs_filepath2
    big_observable2 = np.load(big_filepath2)
    big_observable2 = big_observable2[:,2]

    big_observable = abs(big_ljet_phi - big_observable2)

    title = r'\Delta\phi Between Z, Leading Jet'


#delta phi between Z and subleading jet 
if obs == 'z_sljet_delta_phi':
    z_phi = np.zeros((N,1))
    sl_jet_phi = np.zeros(N)

    for i in range(N):
        sl_jet_phi[i] = np.arctan2(sl_jet[i]['py'],sl_jet[i]['px'])
        z_phi[i] = np.arctan2(data[i]['Particle_py'][np.where(data[i]['Particle_status'] == 22)[0]],data[i]['Particle_px'][np.where(data[i]['Particle_status'] == 22)[0]])

    z_phi = z_phi[:,0]

    observable = abs(zphi - sl_jet_phi)

    big_sljet_phi = big_observable[:,2]
    big_filepath2 = args.big_obs_filepath2
    big_observable2 = np.load(big_filepath2)
    big_observable2 = big_observable2[:,2]

    big_observable = abs(big_sljet_phi - big_observable2)

    title = r'\Delta\phi Between Z, Subleading Jet'


#delta phi between leading and subleading jet 
if obs == 'ljet_sljet_delta_phi':
    l_jet_phi = np.zeros(N)
    sl_jet_phi = np.zeros(N)

    for i in range(N):
        sl_jet_phi[i] = np.arctan2(sl_jet[i]['py'],sl_jet[i]['px'])
        l_jet_phi[i] = np.arctan2(l_jet[i]['py'],l_jet[i]['px'])

    observable = abs(l_jet_phi - sl_jet_phi)

    big_ljet_phi = big_observable[:,2]
    big_filepath2 = args.big_obs_filepath2
    big_observable2 = np.load(big_filepath2)
    big_observable2 = big_observable2[:,2]

    big_observable = abs(big_ljet_phi - big_observable2)

    title = r'\Delta\phi Between Leading, Subleading Jets'


#delta eta between Z and leading jet 
if obs == 'z_ljet_delta_eta':
    z_eta = np.zeros((N,1))
    l_jet_eta = np.zeros(N)

    for i in range(N):
        l_jet_eta[i] = np.arctanh(l_jet[i]['pz']/l_jet[i]['E'])
        z_eta[i] = np.arctanh((data[i]['Particle_pz'][np.where(data[i]['Particle_status'] == 22)[0]]/data[i]['Particle_energy'][np.where(data[i]['Particle_status'] == 22)[0]])) 
    
    z_eta = z_eta[:,0]

    observable = abs(z_eta - l_jet_eta)

    big_ljet_eta = big_observable[:,1]
    big_filepath2 = args.big_obs_filepath2
    big_observable2 = np.load(big_filepath2)
    big_observable2 = big_observable2[:,1]

    big_observable = abs(big_ljet_eta - big_observable2)

    title = r'\Delta\eta Between Z, Leading Jet'


#delta eta between Z and subleading jet 
if obs == 'z_sljet_delta_eta':
    z_eta = np.zeros((N,1))
    sl_jet_eta = np.zeros(N)

    for i in range(N):
        sl_jet_eta[i] = np.arctanh(sl_jet[i]['pz']/sl_jet[i]['E'])
        z_eta[i] = np.arctanh((data[i]['Particle_pz'][np.where(data[i]['Particle_status'] == 22)[0]]/data[i]['Particle_energy'][np.where(data[i]['Particle_status'] == 22)[0]])) 
    
    z_eta = z_eta[:,0]

    observable = abs(z_eta - sl_jet_eta)

    big_sljet_eta = big_observable[:,1]
    big_filepath2 = args.big_obs_filepath2
    big_observable2 = np.load(big_filepath2)
    big_observable2 = big_observable2[:,1]

    big_observable = abs(big_sljet_eta - big_observable2)

    title = r'\Delta\eta Between Z, Subleading Jet'


#delta eta between leading and subleading jet 
if obs == 'ljet_sljet_delta_eta':
    l_jet_eta = np.zeros(N)
    sl_jet_eta = np.zeros(N)

    for i in range(N):
        sl_jet_eta[i] = np.arctanh(sl_jet[i]['pz']/sl_jet[i]['E'])
        l_jet_eta[i] = np.arctanh(l_jet[i]['pz']/l_jet[i]['E'])

    observable = abs(l_jet_eta - sl_jet_eta)

    big_ljet_eta = big_observable[:,1]
    big_filepath2 = args.big_obs_filepath2
    big_observable2 = np.load(big_filepath2)
    big_observable2 = big_observable2[:,1]

    big_observable = abs(big_ljet_eta - big_observable2)

    title = r'\Delta\eta Between Leading, Subleading Jets'


if obs == 'n_jets':
    observable = n_jets

    title = f'Number of Jets Above {jet_pt_cutoff} GeV'

print('Observable Computed Successfully!')

###################################################################################################################################################
#plotting the observable

logscale = args.logscale
quantity = title
numbins = args.numbins
logbins = args.logbins
if logbins == True:
    minbin = args.minbins

Weight = event_weight
bigdata_weight = big_event_weight
bigdata_observable = big_observable
hardreweight = hardprocess_event_weight
showeredreweight = showered_event_weight
hadronreweight = hadronization_event_weight


#create canvas
c1 = ROOT.TCanvas("c1", "Canvas", 800, 600)

#setup all the histograms
if logbins == False:
    hist0 = ROOT.TH1D("Big Data", "My Histogram", numbins, min(bigdata_observable) - min(bigdata_observable) / 100, 
                     max(bigdata_observable) + max(bigdata_observable) / 100)
    
    hist = ROOT.TH1D("Original", "My Histogram", numbins, min(bigdata_observable) - min(bigdata_observable) / 100, 
                     max(bigdata_observable) + max(bigdata_observable) / 100)
    
    hist2 = ROOT.TH1D("Hard Process Reweight", "My Histogram", numbins, min(bigdata_observable) - min(bigdata_observable) / 100, 
                      max(bigdata_observable) + max(bigdata_observable) / 100)
    
    hist3 = ROOT.TH1D("Showered Reweight", 'My Histogram', numbins, min(bigdata_observable) - min(bigdata_observable) / 100, 
                      max(bigdata_observable) + max(bigdata_observable) / 100)
    
    hist4 = ROOT.TH1D('Hadronization Reweight', 'My Histogram', numbins, min(bigdata_observable) - min(bigdata_observable) / 100, 
                      max(bigdata_observable) + max(bigdata_observable) / 100)
elif logbins == True:
    log_bins = np.logspace(np.log10(min(bigdata_observable) - min(bigdata_observable) / 100), np.log10(max(bigdata_observable) + max(bigdata_observable) / 100), 
                           numbins + 1)

    pruned_bins = [log_bins[0]]
    for b in log_bins[1:]:
        if b - pruned_bins[-1] >= minbin:
            pruned_bins.append(b)
            
    log_bins_array = array('d', pruned_bins)

    hist0 = ROOT.TH1D("Big Data", "My Histogram", len(log_bins_array)-1, log_bins_array)
    
    hist = ROOT.TH1D("Original", "My Histogram", len(log_bins_array)-1, log_bins_array)
    
    hist2 = ROOT.TH1D("Hard Process Reweight", "My Histogram", len(log_bins_array)-1, log_bins_array)
    
    hist3 = ROOT.TH1D("Showered Reweight", 'My Histogram', len(log_bins_array)-1, log_bins_array)
    
    hist4 = ROOT.TH1D('Hadronization Reweight', 'My Histogram', len(log_bins_array)-1, log_bins_array)

for value, weight2 in zip(bigdata_observable, bigdata_weight):
    hist0.Fill(value, weight2)  # Fill with value and weight

for value, weight2 in zip(observable, Weight):
    hist.Fill(value, weight2)  # Fill with value and weight

for value, weight2 in zip(observable, hardreweight):
    hist2.Fill(value, weight2)  # Fill with value and weight

for value, weight2 in zip(observable, showeredreweight):
    hist3.Fill(value, weight2)  # Fill with value and weight

for value, weight2 in zip(observable, hadronreweight):
    hist4.Fill(value, weight2)  # Fill with value and weight

ratio0 = hist0.Clone()
ratio0.Divide(hist0)
ratio0.SetLineColor(ROOT.kMagenta)
ratio0.SetStats(0)

ratio = hist.Clone()
ratio.Divide(hist0)
ratio.SetLineColor(ROOT.kBlack)
ratio.SetStats(0)

ratio2 = hist2.Clone()
ratio2.Divide(hist0)
ratio2.SetLineColor(ROOT.kRed)
ratio2.SetStats(0)
ratio2.SetLineWidth(2)

ratio3 = hist3.Clone()
ratio3.Divide(hist0)
ratio3.SetLineColor(ROOT.kBlue)
ratio3.SetStats(0)
ratio3.SetLineWidth(2)

ratio4 = hist4.Clone()
ratio4.Divide(hist0)
ratio4.SetLineColor(ROOT.kGreen)
ratio4.SetStats(0)
ratio4.SetLineWidth(2)

#top plot
pad1 = ROOT.TPad("pad1","pad1" ,0 ,0.3 ,1 ,1)
if logscale == True:
    pad1.SetLogy(True)
else:
    pad1.SetLogy(False)
pad1.Draw()
pad1.cd()
hist0.Draw("h")
hist.Draw("same")
hist2.Draw("same")
hist3.Draw('same')
hist4.Draw('same')

hist0.SetLineColor(ROOT.kMagenta)
hist0.SetLineWidth(2)

hist.SetLineColor(ROOT.kBlack)
hist.SetLineWidth(2)

hist2.SetLineColor(ROOT.kRed)
hist2.SetLineWidth(2)

hist3.SetLineColor(ROOT.kBlue)
hist3.SetLineWidth(2)

hist4.SetLineColor(ROOT.kGreen)
hist4.SetLineWidth(2)

hist0.SetStats(0)

legend = ROOT.TLegend(0.65 ,0.70 ,0.85 ,0.85)
legend.AddEntry(hist0 ,"10M Sample")
legend.AddEntry(hist ,"10k Sample")
legend.AddEntry(hist2 ,"Hard Process Reweight")
legend.AddEntry(hist3 , 'Showered Reweight')
legend.AddEntry(hist4, 'Hadronization Reweight')
legend.SetLineWidth(0)
legend.SetTextSize(0.03)
legend.Draw("same")

ratio0.GetXaxis().SetTitle(quantity)
hist0.GetYaxis().SetTitle("Frequency")


#bottom plot
c1.cd()
pad2 = ROOT.TPad("pad2","pad2" ,0 ,0.05 ,1 ,0.3)
pad2.Draw()
pad2.cd()


pad1.SetBottomMargin(0)
pad2.SetTopMargin(0)
pad2.SetBottomMargin(0.25)

hist0.SetTitle(quantity)
hist0.GetXaxis().SetTitle("")
hist0.GetXaxis().SetTitleSize(0)

ratio0.SetTitle("")
ratio0.GetXaxis().SetLabelSize(0.12)
ratio0.GetXaxis().SetTitleSize(0.12)
ratio0.GetYaxis().SetLabelSize(0.1)
ratio0.GetYaxis().SetTitleSize(0.07)
ratio0.GetYaxis().SetTitle("Reweight / Original")
ratio0.GetYaxis().SetTitleOffset(0.3)
ratio0.GetYaxis().SetRangeUser(0.5 , 1.5)
ratio0.GetYaxis().SetNdivisions(5, ROOT.kTRUE)
ratio0.SetLineWidth(2)


ratio0.Draw()
ratio.Draw('same')
ratio2.Draw('same')
ratio3.Draw('same')
ratio4.Draw('same')


c1.Draw()
c1.Update()
ROOT.gApplication.Run()
