##### Plotting functions for cell-reweighting obs's
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import hist
import numpy as np
import mplhep as hep
hep.style.ROOT
cmap  = cm.tab10.colors 

def get_ratio_unc(num, denom):
    val_num = num.values()
    val_denom = denom.values()
    var_num = num.variances()
    var_denom = denom.variances()
    ratio = np.divide(val_num, val_denom, where=val_denom!=0, out=np.ones_like(val_denom))

    rel_var = np.divide(var_num, val_num**2, where=val_num!=0) + np.divide(var_denom, val_denom**2, where=val_denom!=0)
    ratio_var = (ratio**2)*rel_var
    return ratio, ratio_var

def plot_obs_same_proc(quantity, numbins, xmin, xmax, obs, weights_orig, rw_arrays, fracs, sel=None, ymin=1e-4, ymax=1e0, logy=True):
    if sel is None:
        sel  = np.ones_like(weights_orig, dtype=bool)
    sel_weights = weights_orig[sel]
    cmap  = cm.tab10.colors 
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
    for i, R in enumerate(rw_arrays.keys()):
        weights = rw_arrays[R][sel]
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
        hep.histplot(ratio, bins=bin_edges, ax=rax,  color=cmap[i], histtype='errorbar') #yerr = np.sqrt(ratio_unc),
        hep.histplot(h, ax=ax, label = f"{round(fracs[i], 1)}% RW (R={R} GeV)", color=cmap[i])
    hep.histplot(np.ones_like(ratio), bins=bin_edges, ax=rax, yerr = np.sqrt(ratio_unc1), color='black')
    hep.histplot(h_orig, ax=ax, label = "Original", color='black')
    rax.set_xlabel(rf"${quantity}$")
    rax.set_ylabel("Reweight/Orig")
    ax.set_title("TTbar")
    ax.set_ylabel(rf"d$\sigma/d{quantity}$")
    ax.set_xlabel(None)
    rax.set_ylim(0.85, 1.15)
    rax.set_xlim(xmin, xmax)
    if logy:
        ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    
    ax.legend()
    plt.subplots_adjust(hspace=0.05)
    plt.show()