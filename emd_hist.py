#import useful stuff
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

#check number of cores
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

#import arguments
parser = argparse.ArgumentParser()

parser.add_argument("--distmatrix", type=str, required = True, help = "Path to distance matrix. Must be a csv file")
parser.add_argument("--distmatrix2", type=str, required = False, help = "Path to 2nd distance matrix (if needed). Must be a csv file")
parser.add_argument("--distmatrix3", type=str, required = False, help = "Path to 3rd distance matrix (if needed). Must be a csv file")
parser.add_argument("--numbins", type = int, required = True, help = "Number of bins in histogram")
parser.add_argument("--logscale", action='store_true', help = "Is y axis of plot log scaled or no?")
parser.add_argument("--xmin", type = int, required = False, help = "Lower limit of x range on plot")
parser.add_argument("--xmax", type = int, required = False, help = "Upper limit of x range on plot")
parser.add_argument("--output_path", type = str, required = True, help = "Output file path for histogram")
parser.add_argument("--label1", type = str, required = True, help = 'Label for "distmatrix" dataset')
parser.add_argument("--label2", type = str, required = False, help = 'Label for "distmatrix2" dataset. Required if submitting multiple distance matrices')
parser.add_argument("--label3", type = str, required = False, help = 'Label for "distmatrix3" dataset. Required if submitting multiple distance matrices')


args = parser.parse_args()

print('Starting data preprocessing and plotting...')

#load in distance matrices and remove repeated distances and zeros
filepath = args.distmatrix
dist2neg = np.loadtxt(filepath, delimiter=',')
dist2neg_flat = []
for row in dist2neg:
    zero_idx = np.where(np.isclose(row, 0))[0][0]       # find index of zero in the row
    right_values = row[zero_idx+1:]           # slice everything to the right of zero
    dist2neg_flat.extend(right_values.tolist())      # add to result list
print('First distance matrix preprocessed!')

if args.distmatrix2 is not None:
    filepath2 = args.distmatrix2
    dist2neg2 = np.loadtxt(filepath2, delimiter=',')
    dist2neg2_flat = []
    for row in dist2neg2:
        zero_idx = np.where(np.isclose(row, 0))[0][0]       # find index of zero in the row
        right_values = row[zero_idx+1:]           # slice everything to the right of zero
        dist2neg2_flat.extend(right_values.tolist())      # add to result list   
    print('Second distance matrix preprocessed!')

if args.distmatrix3 is not None:
    filepath3 = args.distmatrix3
    dist2neg3 = np.loadtxt(filepath3, delimiter=',')
    dist2neg3_flat = []
    for row in dist2neg3:
        zero_idx = np.where(np.isclose(row, 0))[0][0]       # find index of zero in the row
        right_values = row[zero_idx+1:]           # slice everything to the right of zero
        dist2neg3_flat.extend(right_values.tolist())      # add to result list 
    print('Third distance matrix preprocessed!')

#bin size if no xmin or xmax and only 1 distance matrix
if args.distmatrix2 is None and args.distmatrix3 is None:
   bins = np.linspace(np.min(dist2neg_flat), np.max(dist2neg_flat), args.numbins)

#bin size if xmin and xmax are not given
if args.distmatrix2 is not None or args.distmatrix3 is not None:
    all_matrices = [dist2neg_flat]
    if args.distmatrix2 is not None:
        all_matrices.append(dist2neg2_flat)
    if args.distmatrix3 is not None:
        all_matrices.append(dist2neg3_flat)
    maximum = np.max([np.max(m) for m in all_matrices]) * 1.05
    minimum = np.min([np.min(m) for m in all_matrices]) * 0.95
    bins = np.linspace(minimum, maximum, args.numbins)
print('Appropriate bins for histogram created. Now plotting')

#modifying bins if xmin and xmax are given
if args.xmin is not None and args.xmax is not None:
    plt.xlim(args.xmin, args.xmax)
    bins = np.linspace(args.xmin, args.xmax, args.numbins)

if (args.xmin is None) != (args.xmax is None):
    print('WARNING: BOTH XMIN AND XMAX ARE NEEDED TO APPLY BOUNDS ON X AXIS')



plt.hist(dist2neg_flat, bins = bins, histtype = 'step', color = 'blue', label = args.label1)
if args.distmatrix2 is not None:
    plt.hist(dist2neg2_flat, bins = bins, histtype = 'step', color = 'red', label = args.label2)
if args.distmatrix3 is not None:
    plt.hist(dist2neg3_flat, bins = bins, histtype = 'step', color = 'green', label = args.label3)


plt.title('Histogram of EMDs')
plt.xlabel('EMD')
plt.legend()

if args.logscale:
    plt.yscale('log')

print('Histogram plotting complete!')

plt.savefig(f"{args.output_path}/EMD_hist.png")  
