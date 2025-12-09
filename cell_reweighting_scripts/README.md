# 1. Reweighting events without trees

Suppose you have a dataset with N total events, of which M are negatively weighted. You should have a M x N distance matrix precomputed. This array should be called to 
define the variable original_dist2neg. The event weights should be called to define the variable event_weight. The function cell_reweight will carry out the reweighting
for a given radius and will return the reweighted weights. 

# 2. Reweighting events with trees

Suppose computing the M x N distance matrix is too expensive. You can still do the reweighting with vantage point (vp) trees. First start with the make_vptree.py 
script. You need to modify the path to the points variables. This leads back to the hard process, showered, and hadronization events. If you want to change the metric
used for reweighting, you'll have to modify the calc_emds function. The changeable parameters are R, beta, and norm. Doing this will make a vp tree for a specific 
dataset for a specific metric form. 

After this to do the reweighting, you'll have to use the 100k_cell_reweighting.py script. Once again, change the paths for the points variables to lead back to the hard
process, showered, and hadronization points. Next, you have to modify the calc_emds function so that it matches the settings for the vp tree you are using. For example,
if your vp tree was made with R = 20, beta = 2, norm = False then when you do the reweighting, the calc_emds function needs to have the same settings. Running this 
script will do the reweighting and will return the new weights and some useful plots. 

Using 100k_cell_reweighting.py can be slow depending on the chosen radius. Therefore its recommended to submit the job to SLURM using reweight_100k.sh. You'll have to 
change the job name, memory per node, time limit, and standard output/error log paths. Then run the python script with whatever inputs you want. 

# 3. Plotting Observables

When the reweighting is done and you have the reweighted distance matrices, you can plot them with the plot_observables script. You will have to change the paths leading to all the observables and reweighted arrays. With this script, you can make observables two ways. The first is with the 10M sample as reference and the other with the original 100k sample as reference. 
