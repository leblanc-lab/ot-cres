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
parser.add_argument("--distmatrix2", type=str, required = True, help = "Path to 2nd distance matrix. Must be a csv file")
parser.add_argument("--numbins", type = int, required = True, help = "Number of bins in histogram")
parser.add_argument("--output_path", type = str, required = True, help = "Output file path for histogram")
parser.add_argument("--label1", type = str, required = True, help = 'Label for "distmatrix" dataset')
parser.add_argument("--label2", type = str, required = True, help = 'Label for "distmatrix2" dataset. Required if submitting multiple distance matrices')

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


filepath2 = args.distmatrix2
dist2neg2 = np.loadtxt(filepath2, delimiter=',')
dist2neg2_flat = []
for row in dist2neg2:
    zero_idx = np.where(np.isclose(row, 0))[0][0]      # find index of zero in the row
    right_values = row[zero_idx+1:]           # slice everything to the right of zero
    dist2neg2_flat.extend(right_values.tolist())      # add to result list

plt.hist2d(dist2neg_flat, dist2neg2_flat, bins=args.numbins, cmap='viridis', norm = 'log')
plt.colorbar(label='Counts')
plt.xlabel(args.label1)
plt.ylabel(args.label2)
plt.title('EMD Correlation Histogram')

print('Plotting complete!')

plt.savefig(f"{args.output_path}/EMD_corr_hist2d.png")