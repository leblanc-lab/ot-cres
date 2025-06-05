#import useful stuff
import numpy as np
import matplotlib.pyplot as plt
import os
import pyhepmc
import argparse

#check number of cores
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

#import arguments
parser = argparse.ArgumentParser()

parser.add_argument("--hard_distmatrix", type=str, required = True, help = "Path to hard process distance matrix. Must be a csv file")
parser.add_argument("--showered_distmatrix", type = str, required = True, help = "Path to showered distance matrix. Must be a csv file")
parser.add_argument("--hadron_distmatrix", type = str, required = True, help = "Path to hadronization distance matrix. Must be a csv file")
parser.add_argument("--max_radius", type = float, required = True, help = "Max cell radius during reweighting. Must be a positive number")
parser.add_argument("--weight_filepath", type=str, required=True, help="Path to the hepmc file of weights")
parser.add_argument("--numbins", type = int, required = True, help = "Number of bins in histogram")
parser.add_argument("--N", type = int, required = True, help = "Number of events to look at")

args = parser.parse_args()
N = args.N
#-----------------------------------------------------------------------------------------------------
#read hepmc file of mc data and store event weights
xsec = np.array([])

with pyhepmc.open(args.weight_filepath) as f:
    for i in f:
        xsec = np.append(xsec, i.weights[0])

print(f'There are {len(np.where(xsec < 0)[0])} negative weighted events out of {len(xsec)} events')

#define event weight
weight = 1.455 * 10**4
event_weight = xsec * weight


#make array of negative weighted events
neg_events = []

for i in range(len(event_weight)):
    if event_weight[i] < 0:
        neg_events.append(i)
        
#-----------------------------------------------------------------------------------------------------
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


print('Hard process reweighting completed')
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



print('Showered reweighting completed')
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


print('Hadronization reweighting completed')
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
#Print and plot some results

#Print reweighted fraction

print(1 - len(np.where(hardprocess_event_weight < 0)[0]) / len(neg_events), f'of negative weights are reweighted for hard process events at {max_radius} GeV cell radius')
print(1 - len(np.where(showered_event_weight < 0)[0]) / len(neg_events), f'of negative weights are reweighted for showered events at {max_radius} GeV cell radius')
print(1 - len(np.where(hadronization_event_weight < 0)[0]) / len(neg_events), f'of negative weights are reweighted for hadronized events at {max_radius} GeV cell radius')

#Plot cell radii histograms
numbins = args.numbins

plt.figure()
plt.hist(hardprocess_cell_radius, bins = numbins, histtype = 'step');
plt.xlabel('Cell Radius (GeV)');
plt.title('Histogram of Cell Radii (Hard Process)');
plt.savefig("hardprocess_cellradius_histogram.png")

plt.figure()
plt.hist(showered_cell_radius, bins = numbins, histtype = 'step');
plt.xlabel('Cell Radius (GeV)');
plt.title('Histogram of Cell Radii (Showered)');
plt.savefig("showered_cellradius_histogram.png")

plt.figure()
plt.hist(hadronization_cell_radius, bins = numbins, histtype = 'step');
plt.xlabel('Cell Radius (GeV)');
plt.title('Histogram of Cell Radii (Hadronized)');
plt.savefig("hadronization_cellradius_histogram.png")

print('All histograms generated and saved')
