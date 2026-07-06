##### Plotting functions for cell-reweighting obs's
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import hist
import numpy as np
import mplhep as hep
import pandas as pd
import os
import re
from matplotlib.colors import LinearSegmentedColormap as lsc
from scipy.optimize import curve_fit
import awkward as ak

hep.style.ROOT

def clean_filename(s):
    s = s.replace('$', '')                # keep content, drop delimiters
    s = re.sub(r'\\', '', s)              # remove LaTeX backslashes (\beta → beta)
    s = re.sub(r'\.', 'p', s)             
    s = re.sub(r'[^A-Za-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')

def get_ratio_unc(num, denom):
    val_num = num.values()
    val_denom = denom.values()
    var_num = num.variances()
    var_denom = denom.variances()
    ratio = np.divide(val_num, val_denom, where=val_denom!=0, out=np.ones_like(val_denom))
    rel_var = np.divide(var_num, val_num**2, where=val_num!=0, out=None) + np.divide(var_denom, val_denom**2, where=val_denom!=0, out=None)
    ratio_var = (ratio**2)*rel_var
    return ratio, ratio_var

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

    
def sigmoid(x ,L, x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return (y)
    
def get_fracs(radii, weights, weights_orig):
    fracs = []
    for R, weight in zip(radii, weights):
        # print("orig neg weights ", ak.sum(weights<0))
        # print("n orig weights ", len(weights))
        # print("Frac orgi negative weights ", ak.sum(weights_orig<0)/len(weights))
        frac = (1-ak.sum(weight<0)/ak.sum(weights_orig<0))
        fracs.append(round(frac, 2))
    return fracs
    
def fit_sigmoid(radii, frac, des_frac=25, string=""):
    from scipy.optimize import curve_fit

    print(frac)
    ###initial guess
    if frac[0]>1.:
        frac.insert(0, 0)
        radii.insert(0,0)
    p0 = [max(frac), np.median(np.array(radii)),1]
    popt, pcov = curve_fit(sigmoid, np.array(radii), frac, p0)
    L, x0, k = popt
    if (L/(des_frac-k))-1 > 0:
        x = x0-((1/k)*np.log((L/(des_frac))-1))
        print(f"{string} x value where y = {des_frac}: {x}")
    else:
        print(f"No real solution for y = {des_frac} with sigmoid fit.")
    return popt, pcov
# def plot_cellradius_comp():
def sigmoid_Lfixed(x, x0, k, L=100.0):
    # clip exponent to avoid overflow warnings
    z = np.clip(-k*(x-x0), -700, 700)
    return L / (1.0 + np.exp(z))
def fit_sigmoid_fixedL(radii, fracs, L=100.0, des_frac=25):
    from scipy.optimize import curve_fit

    # initial guesses
    if fracs[0]>1.:
        fracs.insert(0,0)
        radii.insert(0,0)
    p0 = [np.median(np.array(radii)),1]

    # bounds: x0 within radii range (with padding), k positive
    pad = (np.max(radii) - np.min(radii)) if len(radii) > 1 else 10.0
    bounds = ([np.min(radii) - pad, 1e-6], [np.max(radii) + pad, 10.0])

    popt, pcov = curve_fit(lambda x, x0, k: sigmoid_Lfixed(x, x0, k, L=L),
                           radii, fracs, p0=p0, bounds=bounds)

    x0, k = popt
    return (L, x0, k), pcov
def richards(x, x0, k, nu, L=1.0):
    z = -k*(x-x0)
    return L / (1.0 + np.exp(z))**(1.0/nu)
def fit_richards(radii, fracs, L=1.0):
    from scipy.optimize import curve_fit

    x0_guess = np.median(radii)
    k_guess  = 2/np.median(radii)
    nu_guess = 1.0
        # optional: let L float, or fix it to max observed
    if L is None:
        L = fracs.max()
    if fracs[0]>1.:
        fracs.insert(0,0)
        radii.insert(0,0)
   
    pad = (np.max(radii) - np.min(radii)) if len(radii) > 1 else 10.0
    bounds = (
        [np.min(radii)-pad, 1e-3, 0.05],   # x0, k, nu
        [np.max(radii)+pad, 10.0/np.median(radii), 20.0]
    )

    popt, pcov = curve_fit(lambda x, x0, k, nu: richards(x, x0, k, nu, L=L),
                           radii, fracs, p0=[x0_guess, k_guess, nu_guess], bounds=bounds)
    return (*popt, L), pcov
def richards_x_at_y(y, x0, k, nu, L):
    y = float(y)
    if not (0.0 < y < L) or k == 0 or nu <= 0:
        return np.nan
    arg = (L / y)**nu - 1.0
    if arg <= 0:
        return np.nan
    return x0 - (1.0 / k) * np.log(arg)

def removeTitleDuplication(strings):
    histlabels = []
    testStrings = ["EMD ", "Z+jets ", r"$t\bar{t}$ ", "HAD ", r"$\beta=1$"]
    for testString in testStrings:
        isConsistent = True
        for title in strings:
            if title.find(testString) < 0:
                isConsistent = False
        if isConsistent:
            if not (testString == r"$\beta=1$" and ("Z+jets " in histlabels or r"$t\bar{t}$ " in histlabels)):
                for i, title in enumerate(strings):
                    strings[i] = strings[i].replace(testString, "")
                histlabels.append(testString)
            
    return strings, histlabels

def configure_axis(axis, xlabel, ylabel, fontsizeX = 16, fontsizeY = 16, yaxisAlignment = "top"):
    axis.set_xlabel(xlabel, fontsize=fontsizeX, loc="right", multialignment='center',labelpad=14)
    axis.set_ylabel(ylabel, fontsize=fontsizeY, loc=yaxisAlignment, multialignment='center',labelpad=14)
    axis.minorticks_on()
    axis.tick_params(axis="both", which="major", direction='in', length=10, top=True, right=True, bottom=True, left=True, labelsize=16)
    axis.tick_params(axis="both", which="minor", direction='in', length=5, top=True, right=True, bottom=True, left=True, labelsize=16)

def plot_same_rw_all(obs, df, weights_orig, numbins, xmin, xmax, obs_str = "", obs_title = "", sel = None, ymin=1e-4, ymax=1e0, logy=True, title="", channel="Zjets",raxlim=[0.85, 1.15], logx=False, process_title = "", units="", maxNHists = 6):
    cmap  = ["magenta", "red", "blue", "gold", "lime"]
    markerStyles = ['s', 'o', 'v', '^', 'P', '*', 'x', 'd', '1', '2', '3']

    if sel is None:
        sel  = np.ones_like(weights_orig, dtype=bool)
    if logx:
        bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
        axis_o = hist.axis.Variable(bins,name="data",label="orig",)
        axis_rw = hist.axis.Variable(bins,name="data",label="reweighted",)
    else:
        axis_o = hist.axis.Regular(numbins,xmin, xmax,name="data",label="orig",)
        axis_rw = hist.axis.Regular(numbins,xmin, xmax,name="data",label="orig",)
    sel_weights = weights_orig[sel]

    fig, (ax, rax) = plt.subplots(nrows=2,
                            ncols=1,
                            figsize=(8,8),
                            gridspec_kw={"height_ratios": (3, 1)},
                            sharex=True)

    h_orig = hist.Hist(
            axis_o,
            storage=hist.storage.Weight(), 
        )
    h_orig.fill(obs, weight = sel_weights)
    h_orig = h_orig/h_orig.sum(flow=False).value
    hep.histplot(h_orig, ax=ax, label = "Original", color='black')
    cmap  = tuple(tuple(c) for c in plt.cm.plasma(np.linspace(0.1, 1.0, len(df["radius"]))))

    # We don't want to plot everything if there are too many numbers
    delta = 1
    if(len( df["radius"].values) > maxNHists):
        delta  = int( len( df["radius"].values) / maxNHists)

    for i, R in enumerate(df["radius"].values):
        if i%delta != 0:
            continue
        weights = np.array(df.loc[df["radius"]==R, "weights"].squeeze())[sel]
        frac = df.loc[df["radius"]==R, "fraction"].iloc[0]
        h = hist.Hist(
            axis_rw,
            storage=hist.storage.Weight(), 
        )
        h.fill(obs, weight = weights)
        #### Normalize hists
        h = h/h.sum(flow=False).value
        bin_centers = h.axes[0].centers
        bin_edges = h.axes[0].edges
        ratio, ratio_unc = get_ratio_unc(h, h_orig)
        ratio1, ratio_unc1 = get_ratio_unc(h_orig, h_orig)

        hep.histplot(ratio, bins=bin_edges, ax=rax, histtype='errorbar', yerr = False, marker=markerStyles[i%len(markerStyles)], color =cmap[i], markersize=8 )

        hep.histplot(h, ax=ax, label = f"{round(frac*100)}% RW (R={round(R, 2)} GeV)", histtype='errorbar', marker=markerStyles[i%len(markerStyles)], color=cmap[i], yerr=False, markersize=8 )
    hep.histplot(np.ones_like(ratio), bins=bin_edges, ax=rax, yerr = False, color='black')

    ax.set_xlabel(None)
    ax.legend(frameon=False)
    ax.text(0.05, 0.95, process_title, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=16)

    rax.set_ylim(raxlim[0], raxlim[1])
    rax.set_xlim(xmin, xmax)
    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    configure_axis(ax, "",  r"$\frac{1}{\sigma}\frac{d\sigma}{d%s}$"%obs_str, fontsizeY=24)
    configure_axis(rax, rf"${obs_str} \ {units}$", "Ratio to Original")
    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
    ax.legend(frameon=False)
    plt.subplots_adjust(hspace=0)
    filename = clean_filename(f"{title}_{obs_title}")
    directory = f"../plots/{channel}"
    if not os.path.exists(directory):
      os.makedirs(directory)
    plt.savefig(f"{directory}/{filename}.pdf", bbox_inches='tight')
    plt.show()

def plot_diff_rw(obs, dataset, weights_orig,  process_title = "", sel=None, title="", channel = "Zjets", rwFrac = 0.5, obsName = ""):
    logy= obs["logy"]
    xmin = obs["xmin"]
    xmax = obs["xmax"]
    nbins = obs["numbins"]
    sel = obs["sel"]

    if obs["logx"]:
        bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
        axis_o = hist.axis.Variable(bins,name="data",label="orig",)
        axis_rw = hist.axis.Variable(bins,name="data",label="reweighted",)
    else:
        axis_o = hist.axis.Regular(nbins,xmin, xmax,name="data",label="orig",)
        axis_rw = hist.axis.Regular(nbins,xmin, xmax,name="data",label="orig",)
    if sel is None:
        sel  = np.ones_like(weights_orig, dtype=bool)
    fig, (ax, rax) = plt.subplots(nrows=2,
                        ncols=1,
                        figsize=(8,8),
                        gridspec_kw={"height_ratios": (3, 1)},
                        sharex=True)
    h_orig = hist.Hist(
        axis_o,
        storage=hist.storage.Weight(), )
    h_orig.fill(obs["obs"], weight = weights_orig[sel])
    h_orig = h_orig/h_orig.sum(flow=False).value
    hep.histplot(h_orig, ax=ax, label = "Original", color='black')

    strings = []
    for i, data in enumerate(dataset):
        strings.append(data["title"])
    strings, histlabels = removeTitleDuplication(strings)

    for i, data in enumerate(dataset):
        cfrac = find_nearest(data["df"]["fraction"], rwFrac)
        if(len(np.array(data["df"].loc[data["df"]["fraction"]==cfrac, "weights"].squeeze())) != len(sel)):
            print("nothing matching the fraction")
            print(cfrac, data["df"]["fraction"], rwFrac)
            continue
        weights = np.array(data["df"].loc[data["df"]["fraction"]==cfrac, "weights"].squeeze())[sel]
        h = hist.Hist(
            axis_rw,
            storage=hist.storage.Weight(), 
        )
        h.fill(obs["obs"], weight = weights)
        #### Normalize hists
        h = h/h.sum(flow=False).value
        bin_centers = h.axes[0].centers
        bin_edges = h.axes[0].edges
        ratio, ratio_unc = get_ratio_unc(h, h_orig)
        ratio1, ratio_unc1 = get_ratio_unc(h_orig, h_orig)
        hep.histplot(np.ones_like(ratio), bins=bin_edges, ax=rax, yerr = np.sqrt(ratio_unc1),  color='black')
        hep.histplot(ratio, bins=bin_edges, ax=rax,  color=data["color"], histtype='errorbar', yerr = False, marker=data["marker"], markersize=8, fillstyle="full")
        hep.histplot(h, ax=ax, label = f"{strings[i]}", color=data["color"], histtype='errorbar', yerr= False, marker=data["marker"], markersize=8, fillstyle="full")

    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
    rax.set_ylim(obs["ratioLim"][0], obs["ratioLim"][1])
    rax.set_xlim(xmin, xmax)
    if logy:
        ax.set_yscale('log')
    if obs["logx"]:
        ax.set_xscale('log')
    ax.set_ylim(obs["ymin"], obs["ymax"])
    ax.set_xlim(xmin, xmax)
    configure_axis(ax, "", r"$\frac{1}{\sigma}\frac{d\sigma}{d%s}$"%obs["name"], fontsizeY=24)
    name = obs["name"]
    units = obs["units"]
    configure_axis(rax, rf"${name} \ {units}$", "Ratio to Original", fontsizeX = 20, yaxisAlignment = "center")
    leg = ax.legend(frameon=False, fontsize=15, loc="upper right", borderpad=0.6)
    plt.subplots_adjust(hspace=0.0)
    ax.text(0.05, 0.95, process_title + "\n" r"$f_{rw}$ = %.2f"%(rwFrac), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=18)
    filename = clean_filename(f"{title}_{obsName}_{rwFrac}")
    directory = f"../plots/{channel}"
    if not os.path.exists(directory):
      os.makedirs(directory)
    print(f"{directory}/{filename}")
    plt.savefig(f"{directory}/{filename}.pdf", bbox_inches='tight')
    return ax, leg

def plot_diff_samples(obs0, obs1, dfs, strings, orig_weights, xmin, xmax, nbins, ymin=1e-5, ymax=3e0, obs_str = "", obs_title = "", process_title = "", sel=None, title="", raxlim=[0.85, 1.15], channel = "Zjets", rwFrac = 0.5, logx=False, logy=True, colors = [], markers = [], units = ""):
    if logx:
        bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
        axis_o = hist.axis.Variable(bins,name="data",label="orig",)
        axis_rw = hist.axis.Variable(bins,name="data",label="reweighted",)
    else:
        axis_o = hist.axis.Regular(nbins,xmin, xmax,name="data",label="orig",)
        axis_rw = hist.axis.Regular(nbins,xmin, xmax,name="data",label="orig",)
    if sel is None:
        sel  = np.ones_like(weights_orig, dtype=bool)
    fig, (ax, rax) = plt.subplots(nrows=2,
                        ncols=1,
                        figsize=(8,8),
                        gridspec_kw={"height_ratios": (3, 1)},
                        sharex=True)
    h_orig = hist.Hist(
        axis_o,
        storage=hist.storage.Weight(), )
    h_orig.fill(obs, weight = weights_orig[sel])
    h_orig = h_orig/h_orig.sum(flow=False).value
    hep.histplot(h_orig, ax=ax, label = "Original", color='black')

    for i, df in enumerate(dfs):
        cfrac = find_nearest(df["fraction"], rwFrac)
        if(len(np.array(df.loc[df["fraction"]==cfrac, "weights"].squeeze())) != len(sel)):
            print("nothing matching the fraction")
            print(cfrac, df["fraction"], rwFrac)
            continue
        weights = np.array(df.loc[df["fraction"]==cfrac, "weights"].squeeze())[sel]
        h = hist.Hist(
            axis_rw,
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
        hep.histplot(ratio, bins=bin_edges, ax=rax,  color=colors[i], histtype='errorbar', yerr = False, marker=markers[i], markersize=8, fillstyle="full")
        hep.histplot(h, ax=ax, label = f"{strings[i]}", color=colors[i], histtype='errorbar', yerr= False, marker=markers[i], markersize=8, fillstyle="full")

    configure_axis(rax, rf"${obs_str} \ {units}$", "Ratio to Original")
    configure_axis(ax, "", r"$\frac{1}{\sigma}\frac{d\sigma}{d%s}$"%obs_str)
    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
    rax.set_ylim(raxlim[0], raxlim[1])
    rax.set_xlim(xmin, xmax)
    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.legend(frameon=False, fontsize=12, loc="upper right", borderpad=1.0)
    plt.subplots_adjust(hspace=0.0)
    ax.text(0.05, 0.95, process_title + "\nReweight fraction: %.2f"%(rwFrac), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=16)
    filename = clean_filename(f"{title}_{obs_title}_{rwFrac}")
    directory = f"../plots/{channel}"
    if not os.path.exists(directory):
      os.makedirs(directory)
    print(f"{directory}/{filename}")
    plt.savefig(f"{directory}/{filename}.pdf", bbox_inches='tight')


def plot_radii(dataset, directory, comparison, comparisonDict, process_title = "", maxX = 0):
    fig, ax = plt.subplots()
    maxCellRadius = comparisonDict[comparison]["maxCellRadius"]
    strings = []
    for i, data in enumerate(dataset):
        strings.append(data["title"])
    strings, histlabels = removeTitleDuplication(strings)
    
    for i, data in enumerate(dataset):
        frac = data["df"]["fraction"].values
        radii = data["df"]["radius"].values
        ax.scatter(radii, frac, color=data["color"], label = strings[i], marker=data["marker"])
        popt, pcov = fit_richards(radii, frac)
        x = np.arange(-5,np.max(radii)*1.5)
        if(max(radii) < 20):
          x = np.arange(0,np.max(radii)*1.5*10000)/10000
        frac=0.75
        print(f"{strings[i]} R for {frac} RW ", round(richards_x_at_y(frac, *popt), 3))
        frac=0.5
        print(f"{strings[i]} R for {frac} RW ", round(richards_x_at_y(frac, *popt), 3))
        frac=0.25
        print(f"{strings[i]} R for {frac} RW ", round(richards_x_at_y(frac, *popt), 3))
        ax.plot(x, richards(x, *popt), color=data["color"], linestyle='-')

    ax.text(0.05, 0.95, process_title, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=14)
    if maxX == 0:
        plt.xlim(0, maxCellRadius+(maxCellRadius/6.))
    else:
        plt.xlim(0, maxX)
    configure_axis(ax, "Max cell radius", r"$f_{rw}$")
    plt.ylim(0, 1)
    plt.legend(frameon=False, fontsize=14, loc="lower right", borderpad=1.0)
    filename = clean_filename(f"radii_{comparison}")
    if not os.path.exists(directory):
      os.makedirs(directory)
    print(f"{directory}/{filename}")
    plt.savefig(f"{directory}/{filename}.pdf", bbox_inches='tight')
    plt.clf()

def plot_dilution(neg_percents, dilutions, dataset, title, process_title = ""):
    fig, ax = plt.subplots()
    strings = []
    for i, data in enumerate(dataset):
        strings.append(data["title"])
    strings, histlabels = removeTitleDuplication(strings)
    
    for neg_percent, dilution, data, label in zip(neg_percents, dilutions, dataset, strings):
        ax.plot(neg_percent, dilution, color = data["color"], marker = data["marker"], label = label, linestyle = '--')

    plt.xlim(0, 1)
    
    configure_axis(ax, r"$f_{rw}$", r"$\frac{1}{f_{ESS}}$", fontsizeY = 24)
    ax.legend(frameon=False, fontsize=14, loc="lower left", borderpad=1.0)
    plt.axhline(1, color = 'black', linestyle = '--', label = 'Fully positive, uniform sample')
    ax.text(0.85, 0.95, process_title, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top', fontsize=14)

    
    filename = clean_filename(f"dilution_{title}")
    directory = f"../plots/dilution"
    if not os.path.exists(directory):
      os.makedirs(directory)
    print(f"{directory}/{filename}")
    plt.savefig(f"{directory}/{filename}.pdf", bbox_inches='tight')
    plt.clf()

def plot_variance(neg_percents, variances, dataset, title, process_title = ""):
    fig, ax = plt.subplots()
    strings = []
    for i, data in enumerate(dataset):
        strings.append(data["title"])
    strings, histlabels = removeTitleDuplication(strings)
    
    for neg_percent, variance, data, label in zip(neg_percents, variances, dataset, strings):
        ax.plot(neg_percent, variance, color = data["color"], marker = data["marker"], label = label, linestyle = '--')

    plt.xlim(0, 1)
    
    configure_axis(ax, r"$f_{rw}$", r"Variance")
    ax.text(0.05, 0.95, process_title, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=14)
    ax.legend(frameon=False, fontsize=14, loc="lower left", borderpad=1.0)
    plt.axhline(0, color = 'black', linestyle = '--', label = 'Fully positive, uniform sample')

    filename = clean_filename(f"variance_{title}")
    directory = f"../plots/dilution"
    if not os.path.exists(directory):
      os.makedirs(directory)
        
    print(f"{directory}/{filename}")
    plt.savefig(f"{directory}/{filename}.pdf", bbox_inches='tight')
    plt.clf()


def plot_xmd(fractions, dataset, title, cross_sections, process_title = "", ):
    fig, ax = plt.subplots()
    strings = []
    for i, data in enumerate(dataset):
        strings.append(data["title"])
    strings, histlabels = removeTitleDuplication(strings)
    
    for fraction, data, label, cross_section in zip(fractions, dataset, strings, cross_sections):
        ax.plot(fraction, data['xmd']/cross_section, color = data['color'], marker = data['marker'], label = label, linestyle = data["line"])

    plt.xlim(0, 1)
    plt.ylim(0)

    configure_axis(ax, r"$f_{rw}$", r"$\Sigma MD$ / $\sigma$")
    if len(histlabels) and (histlabels[0]).find("beta")<0:
        ax.text(0.08, 0.95, histlabels[0], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=14)
    ax.legend(frameon=False, fontsize=14, loc="upper left", borderpad=1.7)

    filename = clean_filename(f"xmd_{title}")
    directory = f"../plots/XMDs"
    if not os.path.exists(directory):
      os.makedirs(directory)
    print(f"{directory}/{filename}")
    plt.savefig(f"{directory}/{filename}.pdf", bbox_inches='tight')
    plt.clf()

