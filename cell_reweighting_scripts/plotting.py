##### Plotting functions for cell-reweighting obs's
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import hist
import numpy as np
import mplhep as hep
import re
hep.style.ROOT
cmap  = ["magenta", "red", "blue", "gold", "lime"]
linemap = ["-", "--", "."]
markermap = ['.', '*', '+']

plot_opts = {
        'histtype':'errorbar',
        # 'linestyle': '--', ####if hist type is errorbar, cannot give a linewidth or style
        #'linewidth':1,
        'marker': '.', ####if hist type is step, cannot give a markersize or style
        'markersize': 12,
        'elinewidth': 1,
    }

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

def plot_same_rw_all(obs, df, weights_orig, numbins, xmin, xmax, obs_str = "", sel=None, ymin=1e-4, ymax=1e0, logy=True, title="TTbar", channel="zjets_100k",raxlim=[0.85, 1.15], cmap=cmap, plot_opts=plot_opts):
    if sel is None:
        sel  = np.ones_like(weights_orig, dtype=bool)
    sel_weights = weights_orig[sel]
    cmap  = cm.Dark2.colors 
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
    for i, R in enumerate(df["radius"].values):
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
        hep.histplot(ratio, bins=bin_edges, ax=rax, histtype='step', yerr = np.sqrt(ratio_unc))
        hep.histplot(h, ax=ax, label = f"{round(frac*100)}% RW (R={R} GeV)")
    hep.histplot(np.ones_like(ratio), bins=bin_edges, ax=rax, yerr = np.sqrt(ratio_unc1), color='black')
    rax.set_xlabel(rf"${obs_str}$")
    rax.set_ylabel("Reweight/Orig")
    ax.set_title(title)
    ax.set_ylabel(rf"d$\sigma/d{obs_str}$")
    ax.set_xlabel(None)
    rax.set_ylim(raxlim[0], raxlim[1])
    rax.set_xlim(xmin, xmax)
    if logy:
        ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    ax.legend()
    filename = clean_filename(f"{title}_{obs_str}_allfracs")
    plt.savefig(f"../plots/{channel}/{filename}")
    plt.show()

def plot_diff_rw(obs, dfs, strings, fracs, weights_orig, xmin, xmax, nbins, ymin=1e-5, ymax=1e0, obs_str = "", sel=None, title="", raxlim=[0.75, 1.25], channel = "zjets_100k", cmap=cmap, logx=False, plots_opts=plot_opts):
    if sel is None:
        sel  = np.ones_like(weights_orig, dtype=bool)
    fig, (ax, rax) = plt.subplots(nrows=2,
                        ncols=1,
                        figsize=(8,6),
                        gridspec_kw={"height_ratios": (3, 1)},
                        sharex=True)
    if logx:
        h_orig = hist.Hist(
        hist.axis.Regular(nbins,xmin, xmax,name="data",label="orig",transform=hist.axis.transform.log),
        storage=hist.storage.Weight(), )
    else:
        h_orig = hist.Hist(
        hist.axis.Regular(nbins,xmin, xmax,name="data",label="orig"),
        storage=hist.storage.Weight(), )
    h_orig.fill(obs, weight = weights_orig[sel])
    h_orig = h_orig/h_orig.sum(flow=False).value
    hep.histplot(h_orig, ax=ax, label = "Original", color='black')
    for i, df in enumerate(dfs):
        weights = np.array(df.loc[df["fraction"]==fracs[i], "weights"].squeeze())[sel]
        if logx:
            h = hist.Hist(
            hist.axis.Regular(nbins,xmin, xmax,name="data",label="reweighted",transform=hist.axis.transform.log),
            storage=hist.storage.Weight(), 
            )
        else:
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
        hep.histplot(np.ones_like(ratio), bins=bin_edges, ax=rax, yerr = np.sqrt(ratio_unc1), color='black')
        hep.histplot(ratio, bins=bin_edges, ax=rax,  color=cmap[i], yerr = np.sqrt(ratio_unc), **plot_opts)
        hep.histplot(h, ax=ax, label = f"{round(fracs[i]*100)}% RW {strings[i]}", color=cmap[i], **plot_opts)
    rax.set_xlabel(rf"${obs_str}$")
    rax.set_ylabel("Reweight/Orig")
    ax.set_title(title)
    ax.set_ylabel(rf"d$\sigma/d{obs_str}$")
    ax.set_xlabel(None)
    rax.set_ylim(raxlim[0], raxlim[1])
    ax.set_ylim(ymin, ymax)
    ax.legend()
    ax.set_yscale('log')
    plt.subplots_adjust(hspace=0.05)
    filename = clean_filename(f"{title}_{obs_str}_{fracs[0]}")
    plt.savefig(f"../plots/{channel}/{filename}")
def plot_collection(obs, dfs, string_coll, strings, fracs, weights_orig, xmin, xmax, nbins, ymin=1e-5, ymax=1e0, obs_str = "", sel=None, title="", raxlim=[0.85, 1.15], channel = "zjets_100k", cmap=cmap):
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
    for j, df_coll in enumerate(dfs):
        frac = fracs[j]
        for i, df in enumerate(df_coll):
            weights = np.array(df.loc[df["fraction"]==frac[i], "weights"].squeeze())[sel]
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
            hep.histplot(np.ones_like(ratio), bins=bin_edges, ax=rax, yerr = np.sqrt(ratio_unc1), color='black')
            hep.histplot(ratio, bins=bin_edges, ax=rax,  color=cmap[i], histtype='step', linestyle=linemap[j])
            hep.histplot(h, ax=ax, label = f"{string_coll[j]} {strings[i]}", color=cmap[i], histtype='step', linestyle=linemap[j])
    rax.set_xlabel(rf"${obs_str}$")
    rax.set_ylabel("Reweight/Orig")
    ax.set_title(title)
    ax.set_ylabel(rf"d$\sigma/d{obs_str}$")
    ax.set_xlabel(None)
    rax.set_ylim(raxlim[0], raxlim[1])
    ax.set_ylim(ymin, ymax)
    ax.legend()
    ax.set_yscale('log')
    plt.subplots_adjust(hspace=0.05)
    filename = clean_filename(f"{title}_{obs_str}_{fracs[0]}")
    plt.savefig(f"../plots/{channel}/{filename}")