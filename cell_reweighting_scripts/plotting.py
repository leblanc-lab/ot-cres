##### Plotting functions for cell-reweighting obs's
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import hist
import numpy as np
import mplhep as hep
import os
import re
hep.style.ROOT
cmap  = ["magenta", "red", "blue", "gold", "lime"]

def clean_filename(s):
    s = re.sub(r'\$*?\$', '', s)        # remove LaTeX math
    s = re.sub(r'\.', 'p', s)  ###replace points w/ p
    s = re.sub(r'[^A-Za-z0-9]+', '_', s) # non-alphanumeric → _
    s = re.sub(r'_+', '_', s)            # collapse _
    return s.strip('_')

def get_ratio_unc(num, denom):
    val_num = num.values()
    val_denom = denom.values()
    var_num = num.variances()
    var_denom = denom.variances()
    ratio = np.divide(val_num, val_denom, where=val_denom!=0, out=np.ones_like(val_denom))

    rel_var = np.divide(var_num, val_num**2, where=val_num!=0) + np.divide(var_denom, val_denom**2, where=val_denom!=0)
    ratio_var = (ratio**2)*rel_var
    return ratio, ratio_var

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def plot_same_rw_all(obs, df, weights_orig, numbins, xmin, xmax, obs_str = "", obs_title = "", sel=None, ymin=1e-4, ymax=1e0, logy=True, title="TTbar", channel="zjets_100k",raxlim=[0.85, 1.15]):
    if sel is None:
        sel  = np.ones_like(weights_orig, dtype=bool)
    sel_weights = weights_orig[sel]
    #cmap  = cm.Dark2.colors 
    cmap  = tuple(tuple(c) for c in plt.cm.viridis(np.linspace(0.1, 1.0, len(df["radius"]))))
    markerStyles = ['s', 'o', 'v', '^', 'P', '*', 'x', '1', '2', '3']
    fig, (ax, rax) = plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(8,6),
                            gridspec_kw={"height_ratios": (3, 1)},
                            sharex=True)

    
    h_orig = hist.Hist(
            hist.axis.Regular(numbins,xmin, xmax,name="data",label="TTbar orig",),
            storage=hist.storage.Weight(), 
        )
    h_orig.fill(obs, weight = sel_weights)
    h_orig = h_orig/h_orig.sum(flow=False).value
    hep.histplot(h_orig, ax=ax, label = "Original", color='black')
    cmap  = tuple(tuple(c) for c in plt.cm.viridis(np.linspace(0.1, 1.0, len(df["radius"]))))
    markerStyles = ['s', 'o', 'v', '^', 'P', '*', 'x', 'd', '1', '2', '3']

    # We don't want to plot everything if there are too many numbers
    delta = 1
    maxNHists = 6
    if(len( df["radius"].values) > maxNHists):
        delta  = int( len( df["radius"].values) / maxNHists)

    for i, R in enumerate(df["radius"].values):
        if i%delta != 0:
            continue
        weights = np.array(df.loc[df["radius"]==R, "weights"].squeeze())[sel]
        frac = df.loc[df["radius"]==R, "fraction"].iloc[0]
        h = hist.Hist(
            hist.axis.Regular(numbins,xmin, xmax,name="data",label="TTbar reweighted",),
            storage=hist.storage.Weight(), 
        )
        h.fill(obs, weight = weights)
        #### Normalize hists
        h = h/h.sum(flow=False).value
        bin_centers = h.axes[0].centers
        bin_edges = h.axes[0].edges
        ratio, ratio_unc = get_ratio_unc(h, h_orig)
        ratio1, ratio_unc1 = get_ratio_unc(h_orig, h_orig)
        hep.histplot(ratio, bins=bin_edges, ax=rax, histtype='errorbar', yerr = np.sqrt(ratio_unc), marker=markerStyles[i%len(markerStyles)], color =cmap[i])
        hep.histplot(h, ax=ax, label = f"{round(frac*100)}% RW (R={R} GeV)", histtype='errorbar', marker=markerStyles[i%len(markerStyles)], color=cmap[i])
    hep.histplot(np.ones_like(ratio), bins=bin_edges, ax=rax, yerr = np.sqrt(ratio_unc1), color='black')
    rax.set_xlabel(rf"${obs_str}$", fontsize=12)
    rax.set_ylabel("Ratio to Original", fontsize=12)
    rax.tick_params(axis="both", which="major", direction='in', length=8, top=True, right=True, bottom=True, left=True)
    rax.tick_params(axis="both", which="minor", direction='in', length=4, top=True, right=True, bottom=True, left=True)
    ax.set_title(title)
    ax.set_ylabel(r"$\frac{1}{\sigma}\frac{d\sigma}{d%s}$"%obs_str, fontsize=12)
    ax.set_xlabel(None)
    ax.tick_params(axis="both", which="major", direction='in', length=8, top=True, right=True, bottom=True, left=True)
    ax.tick_params(axis="both", which="minor", direction='in', length=4, top=True, right=True, bottom=True, left=True)
    ax.legend(frameon=False)
    rax.set_ylim(raxlim[0], raxlim[1])
    rax.set_xlim(xmin, xmax)
    if logy:
        ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    ax.legend(frameon=False)
    plt.subplots_adjust(hspace=0.05)
    directory = f"../plots/{channel}"
    if not os.path.exists(directory):
      os.makedirs(directory)
    plt.savefig(f"{directory}/{title}_{obs_str}.png")
    plt.show()

def plot_diff_rw(obs, dfs, strings, weights_orig, xmin, xmax, nbins, ymin=1e-5, ymax=1e0, obs_str = "", obs_title = "", process_title = "", sel=None, title="", raxlim=[0.85, 1.15], channel = "zjets_100k", rwFrac = 0.5):
    if sel is None:
        sel  = np.ones_like(weights_orig, dtype=bool)
    fig, (ax, rax) = plt.subplots(nrows=2,
                        ncols=1,
                        figsize=(8,6),
                        gridspec_kw={"height_ratios": (3, 1)},
                        sharex=True)
    h_orig = hist.Hist(
        hist.axis.Regular(nbins,xmin, xmax,name="data",label="orig",),
        storage=hist.storage.Weight(), )
    h_orig.fill(obs, weight = weights_orig[sel])
    h_orig = h_orig/h_orig.sum(flow=False).value
    hep.histplot(h_orig, ax=ax, label = "Original", color='black')
    cmap  = tuple(tuple(c) for c in plt.cm.viridis(np.linspace(0.1, 1.0, len(dfs))))
    markerStyles = ['s', 'o', 'v', '^', 'P', '*', 'x']

    for i, df in enumerate(dfs):
        cfrac = find_nearest(df["fraction"], rwFrac)
        if(len(np.array(df.loc[df["fraction"]==cfrac, "weights"].squeeze())) != len(sel)):
            print("nothing matching the fraction")
            print(cfrac, df["fraction"], rwFrac)
            continue
        weights = np.array(df.loc[df["fraction"]==cfrac, "weights"].squeeze())[sel]
        h = hist.Hist(
            hist.axis.Regular(nbins,xmin, xmax,name="data",label="reweighted",),
            storage=hist.storage.Weight(), 
        )
        h.fill(obs, weight = weights)
        #### Normalize hists
        h = h/h.sum(flow=False).value
        bin_centers = h.axes[0].centers
        bin_edges = h.axes[0].edges
        ratio, ratio_unc = get_ratio_unc(h, h_orig)
        ratio1, ratio_unc1 = get_ratio_unc(h_orig, h_orig)
        hep.histplot(np.ones_like(ratio), bins=bin_edges, ax=rax, yerr = np.sqrt(ratio_unc1),  color='black')
        hep.histplot(ratio, bins=bin_edges, ax=rax,  color=cmap[i], histtype='errorbar', yerr = np.sqrt(ratio_unc), marker=markerStyles[i] )
        hep.histplot(h, ax=ax, label = f"{round(cfrac*100)}% RW {strings[i]}", color=cmap[i], histtype='errorbar', marker=markerStyles[i])
    rax.set_xlabel(rf"${obs_str}$", fontsize=12)
    rax.set_ylabel("Ratio to Original", fontsize=12)
    rax.tick_params(axis="both", which="major", direction='in', length=8, top=True, right=True, bottom=True, left=True)
    rax.tick_params(axis="both", which="minor", direction='in', length=4, top=True, right=True, bottom=True, left=True)
    rax.set_ylim(raxlim[0], raxlim[1])
    
    ax.set_title(process_title)
    ax.set_ylabel(r"$\frac{1}{\sigma}\frac{d\sigma}{d%s}$"%obs_str, fontsize=12)
    ax.set_xlabel(None)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis="both", which="major", direction='in', length=8, top=True, right=True, bottom=True, left=True)
    ax.tick_params(axis="both", which="minor", direction='in', length=4, top=True, right=True, bottom=True, left=True)
    ax.legend(frameon=False)
    ax.set_yscale('log')
    plt.subplots_adjust(hspace=0.05)
    filename = clean_filename(f"{title}_{obs_title}_{rwFrac}")
    directory = f"../plots/{channel}"
    if not os.path.exists(directory):
      os.makedirs(directory)
    plt.savefig(f"{directory}/{filename}")
