import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt

def make_df(radii, weights_orig, inputpath0, inputpath1 = "gev_bigR.npy", points = "None"):
    weights = []
    for R in radii:
        str_R = str(R).replace(".", "p")
        weight = np.load(inputpath0+str_R+inputpath1)
        if type(points)!=str:
            weight = fix_weight_length(points, weight, weights_orig)
        weights.append(weight)
    fracs = get_fracs(radii, weights, weights_orig)
    df = pd.DataFrame({
        "radius": radii,
        "fraction": fracs,
        "weights": weights
    })
    return df

    
def fix_weight_length(points, weights, weights_orig):
    mask = np.all(points == 0, axis=(1, 2))
    if len(weights) != len(weights_orig):
        rw_weights = weights_orig.copy()
        print("Masking out zero values to fix different lengths: ", mask.shape, rw_weights.shape, weights.shape)
        rw_weights[~mask] = weights
        return rw_weights
    else:
        return weights

def get_fracs(radii, weights, weights_orig):
    fracs = []
    for R, weight in zip(radii, weights):
        # print("orig neg weights ", ak.sum(weights<0))
        # print("n orig weights ", len(weights))
        # print("Frac orgi negative weights ", ak.sum(weights_orig<0)/len(weights))
        frac = (1-ak.sum(weight<0)/ak.sum(weights_orig<0))
        fracs.append(round(frac, 2))
    return fracs

def get_negative_percent(weight, reweights):
    return [1 - len(np.asarray(i)[np.asarray(i)<0])/len(weight[weight<0]) for i in reweights]


def get_dilution(reweights):
    means = [np.mean(np.asarray(i)/14550) for i in reweights]
    meansSquared = [np.mean((np.asarray(i)/14550)**2) for i in reweights]

    return (1 / (np.array(means)**2 / meansSquared))


hpString="HS"
psString="PS"
hadString="HAD"

rjainDir = "/oscar/data/mleblan6/rjain"
#rjainDir = "/Users/jroloff/Work/cellReweighting/plotting/cres-distance"
lhayDir = "/oscar/data/mleblan6/lhay"
#lhayDir = "/Users/jroloff/Work/cellReweighting/plotting/cres-distance"
jmarrinanDir = "/oscar/data/mleblan6/jmarrinan"
#jmarrinanDir = "/Users/jroloff/Work/cellReweighting/plotting/cres-distance"

with open(rjainDir + "/jeppe_new/radii.txt") as f:
    code = f.read()
exec(code)
dataFrames = {}
weights_orig_z = np.load(rjainDir + "/ppzjj_100k/weight_100k.npy")

###ttbar xmds
ps_ttbar_xmd_dist = np.load(lhayDir + '/xmd_outputs/ttbar_100k_xmd_ps.npy')
had_ttbar_xmd_dist = np.load(lhayDir + '/xmd_outputs/ttbar_100k_xmd_had.npy')
hp_ttbar_xmd_dist = np.load(lhayDir + '/xmd_outputs/ttbar_100k_xmd_hp.npy')
jeppe_xmd = np.load(rjainDir + "/hundredK_XMD/jeppe_XMD.npy")
jeppe_xmd_ttbar = np.load(rjainDir + "/hundredK_XMD/ttbar_jeppe_XMD.npy")
#### no ttbar for V1
jeppe_xmdV1 = np.load(rjainDir + "/jeppe_new/zjj_jeppe_NEW_XMD.npy")
jeppe_xmd_ttbarV1 = np.load(lhayDir+ "/xmd_outputs/ttbar_jeppe_NEW_v2_XMD.npy")
#### zjj not finished for V2
jeppe_xmdV2 = np.load(rjainDir + "/jeppe_new_v2/zjj_jeppe_NEW_v2_XMD.npy")
jeppe_xmd_ttbarV2 = np.load(lhayDir + "/xmd_outputs/ttbar_jeppe_NEW_v2_XMD.npy")

jeppe_xmdV3 = np.load(lhayDir + "/xmd_outputs/zjj_jeppe_NEW_v3_XMD.npy")
jeppe_xmd_ttbarV3 = np.load(lhayDir + "/xmd_outputs/ttbar_jeppe_NEW_v3_XMD.npy")

##### Load jets
hp_points = np.load(rjainDir + "/ppzjj_100k/hardprocess_points.npy")

colors = [ "#ff1f2a", "#00a8fc", "#ff9d00", "#b374ff", "#64c704", "#0000ff", "#a60001", "#f65295", "#99dee1", "#9d02d7", "#ded800", "#007823" , "darkslategray" ]
colors2 = tuple(tuple(c) for c in plt.cm.turbo(np.linspace(0.1, 1.0, 10)))
markers = ["o", "v", "^", "s", "X", "h", "*", "p", "d", "D", "P", "8"]
pts = np.load(rjainDir + '/ppzjj_100k/hardprocess_points.npy')[:,:,0]

#### load original weights for ttbar, z+jet
N=10000
weights_orig_10k = np.load(rjainDir + '/ppzjj_100k/weight_100k.npy')[N:2*N]
weights_orig_z = np.load(rjainDir + "/ppzjj_100k/weight_100k.npy")

weight = np.load(rjainDir + '/ppzjj_100k/weight_100k.npy')

#### load original weights for ttbar, z+jet
N=10000
weights_orig_10k = np.load(rjainDir + '/ppzjj_100k/weight_100k.npy')[N:2*N]

#### making the beta=1 hard process dataframe
radii_hp = [5, 7, 9, 9.2, 9.5, 10, 12, 12.5, 12.9, 13.1, 15, 18, 20, 25]
zjet_hp_df = make_df(radii_hp, weight, rjainDir + "/100k_reweight_bigR/100k_hp_emd_reweight_","gev_bigR.npy", points=hp_points)
hp_xmd_dist = np.load(rjainDir + '/hundredK_XMD/hp_XMD_v3.npy')
dataFrames["zjet_hp_df"] = {"process": "Zjets", "df": zjet_hp_df, "xmd":hp_xmd_dist, "title": r"Z+jets $\beta=1$ %s "%hpString, "color": colors[0], "marker": markers[0], "line": '--'}

#### showered
radii_ps_100k =  [5, 9, 10, 12.5, 12.6, 15, 17.2, 20, 25]
zjet_ps_df = make_df(radii_ps_100k, weight, rjainDir + "/100k_reweight_bigR/100k_sho_emd_reweight_","gev_bigR.npy")
sho_xmd_dist = np.load(rjainDir + '/hundredK_XMD/sho_XMD_v3.npy')
dataFrames["zjet_ps_df"] = {"process": "Zjets", "df": zjet_ps_df, "xmd":sho_xmd_dist, "title": r"Z+jets $\beta=1$ %s "%psString, "color": colors[1], "marker": markers[1], "line": '--'}

#### hadronization 
radii_had_100k = [5, 10, 13, 15, 17.2, 20, 25]
zjet_had_df = make_df(radii_had_100k, weight, rjainDir + "/100k_reweight_bigR/100k_had_emd_reweight_","gev_bigR.npy")
# zjet_had_df = make_df(radii_had_100k, weight, "/oscar/data/mleblan6/lhay/reweighted_files/zjet_100k_had/100k_had_emd_reweight_","gev_bigR.npy")
had_xmd_dist = np.load(rjainDir + '/hundredK_XMD/had_XMD_v3.npy')
dataFrames["zjet_had_df"] = {"process": "Zjets", "df": zjet_had_df, "xmd":had_xmd_dist, "title": r"Z+jets $\beta=1$ %s "%hadString, "color": colors[2], "marker": markers[2], "line": '--'}


###### Camille hard process SEMD RW's
hp_semd_reweights = np.load(rjainDir + '/semd_reweight/hardprocess_reweights.npy')
fracs = get_fracs(np.logspace(1,np.log10(200),50), hp_semd_reweights, weight)
zjet_SEMD_p2_hp_df = pd.DataFrame({"radius": np.logspace(1,np.log10(200),50), "fraction": fracs, "weights": hp_semd_reweights.tolist()})
hp_semd_xmd_dist = np.load(rjainDir + '/hundredK_XMD/hp_semd_XMD.npy')
dataFrames["zjet_SEMD_cm"] = {"process": "Zjets", "df": zjet_SEMD_p2_hp_df,"xmd":hp_semd_xmd_dist, "title": "Z+jets SEMD p=2 %s"%hpString, "color": colors[3], "marker": markers[3], "line": '--'}

#### Jeppe RW
jeppe_rw = np.load(rjainDir + "/jeppe_new/zjj_jeppe_reweights.npy")
fracs = get_fracs(zjj_radii, jeppe_rw, weight)
zjet_jeppeV1_df = pd.DataFrame({"radius": zjj_radii, "fraction": fracs, "weights": jeppe_rw.tolist()})
dataFrames["zjet_jeppeV1_df"] = {"process": "Zjets", "df": zjet_jeppeV1_df, "xmd":jeppe_xmdV1,"title": "Z+jets V1 Andersen $et$ $al.$", "color": colors[5], "marker": markers[5], "line": '--'}

#### Jeppe RW OLD
with open(rjainDir + "/jeppe/radii.txt") as f:
    code = f.read()
exec(code)
jeppe_rw = np.load(rjainDir + "/jeppe/jeppe_reweights.npy")
fracs = get_fracs(radii, jeppe_rw, weight)
zjet_jeppeV0_df = pd.DataFrame({"radius": radii, "fraction": fracs, "weights": jeppe_rw.tolist()})
dataFrames["zjet_jeppeV0_df"] = {"process": "Zjets", "df": zjet_jeppeV0_df, "xmd":jeppe_xmd,"title": "Z+jets V0 Andersen $et$ $al.$", "color": colors[7], "marker": markers[7], "line": '--'}

zjj_radii = np.logspace(1,np.log10(350),50)
jeppe_rw = np.load(rjainDir + "/jeppe_new_v2/zjj_jeppe_reweights.npy")
fracs = get_fracs(zjj_radii, jeppe_rw, weight)
zjet_jeppeV2_df = pd.DataFrame({"radius": zjj_radii, "fraction": fracs, "weights": jeppe_rw.tolist()})
dataFrames["zjet_jeppeV2_df"] = {"process": "Zjets", "df": zjet_jeppeV2_df, "xmd":jeppe_xmdV2,"title": "Z+jets V2 Andersen $et$ $al.$", "color": colors[6], "marker": markers[6], "line": '--'}

zjj_radii = np.logspace(0,np.log10(350),50)
jeppe_rw = np.load(rjainDir + "/jeppe_new_v3/zjj_jeppe_reweights.npy")
fracs = get_fracs(zjj_radii, jeppe_rw, weight)
zjet_jeppeV3_df = pd.DataFrame({"radius": zjj_radii, "fraction": fracs, "weights": jeppe_rw.tolist()})
dataFrames["zjet_jeppeV3_df"] = {"process": "Zjets", "df": zjet_jeppeV3_df, "xmd":jeppe_xmdV3,"title": "Z+jets Andersen $et$ $al.$", "color": colors[4], "marker": markers[4], "line": '--'}

##### Rishabh SEMD RW's w/ p=1
radii_hp = np.logspace(-5,-2,50)
hp_rw = np.load(rjainDir + "/p1_semd/hardprocess_reweights.npy")
hp_rw = np.array([fix_weight_length(hp_points, rw, weight) for rw in hp_rw])
fracs = get_fracs(radii_hp, hp_rw, weight)
zjet_SEMD_hp_df = pd.DataFrame({"radius": radii_hp, "fraction": fracs, "weights": hp_rw.tolist()})
hp_p1semd_xmd_dist = np.load(rjainDir + '/hundredK_XMD/hp_p1_semd_XMD.npy')
dataFrames["zjet_SEMD_hp_df"] = {"process": "Zjets", "df": zjet_SEMD_hp_df, "xmd":hp_p1semd_xmd_dist, "title": "Z+jets SEMD p=1 %s"%hpString, "color":colors2[0], "marker": markers[0], "line": '--'}
    
radii_ps =  np.logspace(-4,-2,50)
ps_rw = np.load(rjainDir + "/p1_semd/showered_reweights.npy")
fracs = get_fracs(radii_ps, ps_rw, weight)
zjet_SEMD_ps_df = pd.DataFrame({"radius": radii_ps, "fraction": fracs, "weights": ps_rw.tolist()})
sho_p1semd_xmd_dist = np.load(rjainDir + '/hundredK_XMD/sho_p1_semd_XMD.npy')
dataFrames["zjet_SEMD_ps_df"] = {"process": "Zjets", "df": zjet_SEMD_ps_df, "xmd":sho_p1semd_xmd_dist, "title": "Z+jets SEMD p=1 %s"%psString, "marker": markers[3], "color": colors[8], "line": '--'}

had_rw = np.load(rjainDir + "/p1_semd/hadronization_reweights.npy")
fracs = get_fracs(radii_ps, had_rw, weight)
zjet_SEMD_had_df = pd.DataFrame({"radius": radii_ps, "fraction": fracs, "weights": had_rw.tolist()})
had_p1semd_xmd_dist = np.load(rjainDir + '/hundredK_XMD/had_p1_semd_XMD.npy')
dataFrames["zjet_SEMD_had_df"] = {"process": "Zjets", "df": zjet_SEMD_had_df, "xmd":had_p1semd_xmd_dist, "title": "Z+jets SEMD p=1 %s"%hadString, "color": colors[5], "marker": markers[3], "line": '--'}

#### beta=0
radii_b0 = np.logspace(1.5,3,50)
b0_hp_rw = np.load(rjainDir + "/beta0/hardprocess_reweights_fixed.npy")*14456.19
b0_hp_rw = np.array([fix_weight_length(hp_points, rw, weight) for rw in b0_hp_rw])
#print(b0_hp_rw.shape)
fracs = get_fracs(radii_b0, b0_hp_rw, weight)
zjet_b0_hp_df = pd.DataFrame({"radius": radii_b0[0:26], "fraction": fracs[0:26], "weights": b0_hp_rw.tolist()[0:26]})
hp_beta0_xmd_dist = np.load(rjainDir + '/hundredK_XMD/hp_beta0_XMD.npy')
dataFrames["zjet_b0_hp_df"] = {"process": "Zjets", "df": zjet_b0_hp_df, "xmd":hp_beta0_xmd_dist,"title": r"Z+jets $\beta=0$ %s"%hpString, "marker": markers[4], "color":colors2[5], "line": '--'}

b0_ps_rw = np.load(rjainDir + "/beta0/showered_reweights_fixed.npy")*14456.19
fracs = get_fracs(radii_b0, b0_ps_rw, weight)
zjet_b0_ps_df = pd.DataFrame({"radius": radii_b0[0:29], "fraction": fracs[0:29], "weights": b0_ps_rw.tolist()[0:29]})
sho_beta0_xmd_dist = np.load(rjainDir + '/hundredK_XMD/sho_beta0_XMD.npy')
dataFrames["zjet_b0_ps_df"] = {"process": "Zjets", "df": zjet_b0_ps_df, "xmd":sho_beta0_xmd_dist,"title": r"Z+jets $\beta=0$ %s"%psString,"marker": markers[4], "color": colors2[5], "line": '--'}

b0_had_rw = np.load(rjainDir + "/beta0_reorder/had_beta0_reweights.npy")*14456.19
radii_b0_had = np.logspace(np.log10(50), np.log10(350), 50)
fracs = get_fracs(radii_b0_had, b0_had_rw, weight)
zjet_b0_had_df = pd.DataFrame({"radius": radii_b0_had[0:40], "fraction": fracs[0:40], "weights": b0_had_rw[0:40].tolist()})
had_beta0_xmd_dist = np.load(rjainDir + '/beta0_reorder/had_beta0_XMD_distances.npy')
dataFrames["zjet_b0_had_df"] = {"process": "Zjets", "df": zjet_b0_had_df, "xmd":had_beta0_xmd_dist[0:40],"title": r"Z+jets $\beta=0$ %s"%hadString, "color": colors[6], "marker": markers[5], "line": '--'}


#### beta=infinity
radii_binf = np.logspace(-4,-1,50)
binf_hp_rw = np.load(rjainDir + "/betainf/hardprocess_reweights.npy")
binf_hp_rw = np.array([fix_weight_length(hp_points, rw, weight) for rw in binf_hp_rw])
fracs = get_fracs(radii_binf, binf_hp_rw, weight)
zjet_binf_hp_df = pd.DataFrame({"radius": radii_binf, "fraction": fracs, "weights": binf_hp_rw.tolist()})
hp_betainf_xmd_dist = np.load(rjainDir + '/hundredK_XMD/hp_betainf_XMD.npy')
dataFrames["zjet_binf_hp_df"] = {"process": "Zjets", "df": zjet_binf_hp_df, "xmd":hp_betainf_xmd_dist,"title": r"Z+jets $\beta=\infty$ %s"%hpString, "marker": markers[5], "color":colors2[6], "line": '--'}

binf_ps_rw = np.load(rjainDir + "/betainf/showered_reweights.npy")
fracs = get_fracs(radii_binf, binf_ps_rw, weight)
zjet_binf_ps_df = pd.DataFrame({"radius": radii_binf, "fraction": fracs, "weights": binf_ps_rw.tolist()})
sho_betainf_xmd_dist = np.load(rjainDir + '/hundredK_XMD/sho_betainf_XMD.npy')
dataFrames["zjet_binf_ps_df"] = {"process": "Zjets", "df": zjet_binf_ps_df, "xmd":sho_betainf_xmd_dist,"title": r"Z+jets $\beta=\infty$ %s"%psString, "marker": markers[5], "color": colors2[6], "line": '--'}

binf_had_rw = np.load(rjainDir + "/betainf/hadronization_reweights.npy")*14456.19
fracs = get_fracs(radii_binf, binf_had_rw, weight)
zjet_binf_had_df = pd.DataFrame({"radius": radii_binf, "fraction": fracs, "weights": binf_had_rw.tolist()})
had_betainf_xmd_dist = np.load(rjainDir + '/hundredK_XMD/had_betainf_XMD.npy')
dataFrames["zjet_binf_had_df"] = {"process": "Zjets", "df": zjet_binf_had_df, "xmd":had_betainf_xmd_dist,"title": r"Z+jets $\beta=\infty$ %s"%hadString, "color": colors[7], "marker": markers[6], "line": '--'}

#### z+2jet beta=0.5 dataframes
radii_beta0p5_hp_100k = [10,20,30,40]
zjet_b0p5_hp_df = make_df(radii_beta0p5_hp_100k, weight, jmarrinanDir + "/hp_reweights/100k_hp_emd_reweight_", "gev_Rmax_beta0p5.npy", points=hp_points)
hp_beta0p5_xmd_dist = np.load(rjainDir + '/hundredK_XMD/hp_beta0p5_XMD.npy')
dataFrames["zjet_b0p5_hp_df"] = {"process": "Zjets", "df": zjet_b0p5_hp_df, "xmd":hp_beta0p5_xmd_dist,"title": r"Z+jets $\beta=0.5$ %s"%hpString, "marker": markers[6], "color":colors2[7], "line": '--'}

radii_beta0p5_ps_100k = [25,30,40,50]
zjet_b0p5_ps_df = make_df(radii_beta0p5_ps_100k, weight, jmarrinanDir + "/sho_reweights/100k_shower_emd_reweight_", "gev_Rmax_beta0p5.npy")
sho_beta0p5_xmd_dist = np.load(rjainDir + '/hundredK_XMD/sho_beta0p5_XMD.npy')
dataFrames["zjet_b0p5_ps_df"] = {"process": "Zjets", "df": zjet_b0p5_ps_df, "xmd":sho_beta0p5_xmd_dist,"title": r"Z+jets $\beta=0.5$ %s"%psString, "marker": markers[6], "color": colors2[7], "line": '--'}

radii_beta0p5_had_100k = [10,20,30,33,40,50, 60]
zjet_b0p5_had_df = make_df(radii_beta0p5_had_100k, weight, lhayDir + "/zjet_beta0p5/100k_had_emd_b0p5_reweight_", "gev.npy")
had_beta0p5_xmd_dist = np.load(rjainDir + '/hundredK_XMD/had_beta0p5_XMD.npy')
dataFrames["zjet_b0p5_had_df"] = {"process": "Zjets", "df": zjet_b0p5_had_df, "xmd":had_beta0p5_xmd_dist,"title": r"Z+jets $\beta=0.5$ %s"%hadString, "color": colors2[1], "marker": markers[7], "line": '--'}

#### z+2jet beta=2 dataframes
radii_beta2_hp_100k = np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7])
zjet_b2_hp_df = make_df(radii_beta2_hp_100k, weight, lhayDir + "/zjet_beta2NEW/100k_hp_emd_b2NEW_reweight_", "gev.npy")
hp_beta2_xmd_dist = np.load(rjainDir + '/hundredK_XMD/hp_beta2_XMD.npy')
dataFrames["zjet_b2_hp_df"] = {"process": "Zjets", "df": zjet_b2_hp_df, "xmd":hp_beta2_xmd_dist,"title": r"Z+jets $\beta=2$ %s"%hpString, "color":colors[12], "marker": markers[8], "line": '--'}

radii_beta2_sho_100k = np.array([1.0,1.1,1.2,1.3,1.5,1.6,1.7,1.8])
path = lhayDir + '/zjet_beta2NEW/100k_ps_emd_b2NEW_reweight_'
zjet_b2_ps_df = make_df(radii_beta2_sho_100k, weight, path, "gev.npy")
sho_beta2_xmd_dist = np.load(rjainDir + '/hundredK_XMD/sho_beta2_XMD.npy')[:8]
dataFrames["zjet_b2_ps_df"] = {"process": "Zjets", "df": zjet_b2_ps_df, "xmd":sho_beta2_xmd_dist,"title": r"Z+jets $\beta= 2$ %s"%psString, "color":colors2[9], "marker": markers[8], "line": '--'}

radii_beta2_had_100k= np.array([1.0,1.1,1.2,1.3,1.35,1.4,1.5,1.53,1.6])
path = lhayDir + '/zjet_beta2NEW/100k_had_emd_b2NEW_reweight_'
zjet_b2_had_df = make_df(radii_beta2_had_100k, weight, lhayDir + "/zjet_beta2NEW/100k_had_emd_b2NEW_reweight_", "gev.npy")
had_beta2_xmd_dist = np.load(rjainDir + '/hundredK_XMD/had_beta2_XMD.npy')[:9]
dataFrames["zjet_b2_had_df"] = {"process": "Zjets", "df": zjet_b2_had_df, "xmd":had_beta2_xmd_dist,"title": r"Z+jets $\beta= 2$ %s"%hadString, "color": colors[11], "marker": markers[8], "line": '--'}

#### ttbar samples
weights_orig_ttbar = np.load(lhayDir + "/ttbar_100k/ttbar_weight_100k.npy")
weight = np.load(rjainDir + "/ppzjj_100k/weight_100k.npy")

radii_hp = [5,15,17,18.5,20,22,25,30,100]
ttbar_hp_df = make_df(radii_hp, weights_orig_ttbar, lhayDir + "/reweighted_files/ttbar_hp_100k/100k_hp_emd_reweight_")
dataFrames["ttbar_hp_df"] = {"process": "TTbar", "df": ttbar_hp_df, "xmd":hp_ttbar_xmd_dist,"title": r"$t\bar{t}$ $\beta=1$ %s "%hpString, "color": colors[0], "marker": markers[0], "line": '-'}
#dataFrames["ttbar_hp_df"] = {"process": "TTbar", "df": ttbar_hp_df, "title": r"$t\bar{t}$ %s "%hpString, "color": colors[0], "marker": markers[9]}


#### make rest of ttbar dataframes
radii_ps = [1, 15, 18, 24, 25, 30, 50, 100]
ttbar_ps_df = make_df(radii_ps, weights_orig_ttbar, lhayDir + "/reweighted_files/ttbar_ps_100k/100k_ps_emd_reweight_", "gev_ttbar.npy")
dataFrames["ttbar_ps_df"] = {"process": "TTbar", "df": ttbar_ps_df, "xmd":ps_ttbar_xmd_dist,"title": r"$t\bar{t}$ $\beta=1$ %s "%psString, "color": colors[1], "marker": markers[1], "line": '-'}
#dataFrames["ttbar_ps_df"] = {"process": "TTbar", "df": ttbar_ps_df, "title": r"$t\bar{t}$ %s "%psString, "color": colors[1], "marker": markers[9]}


radii_had = [1, 18, 25, 28, 30, 50, 100]
ttbar_had_df = make_df(radii_had[:-1], weights_orig_ttbar, lhayDir + "/reweighted_files/ttbar_had_100k/100k_had_emd_reweight_", "gev_ttbar.npy")
print("TTbar frac values", ttbar_had_df['fraction'].values)
dataFrames["ttbar_had_df"] = {"process": "TTbar", "df": ttbar_had_df, "xmd":had_ttbar_xmd_dist[:-1], "title": r"$t\bar{t}$ $\beta=1$ %s "%hadString, "color":colors[2], "marker": markers[2], "line": '-'}

#### Jeppe RW TTbar
ttbar_radii = np.logspace(np.log10(200), 3, 50)
jeppe_rw = np.load(rjainDir + "/jeppe/ttbar_jeppe_reweights.npy")*(678.620)
fracs = get_fracs(ttbar_radii, jeppe_rw, weights_orig_ttbar)
print("TTbar jeppe v0 frac values", fracs)
ttbar_jeppe_df = pd.DataFrame({"radius": ttbar_radii, "fraction": fracs, "weights": jeppe_rw.tolist()})
dataFrames["ttbar_jeppeV0_df"] = {"process": "TTbar", "df": ttbar_jeppe_df, "xmd":jeppe_xmd_ttbar,"title": r"$t\bar{t}$ Andersen V0 $et$ $al.$", "color": colors[4], "marker": markers[4], "line": '--'}

#### Jeppe RW TTbar
ttbar_radii = np.logspace(np.log10(200),3,50)
jeppe_rw = np.load(rjainDir + "/jeppe_new/ttbar_jeppe_reweights.npy")*14456.19
fracs = get_fracs(ttbar_radii, jeppe_rw, weights_orig_ttbar)
ttbar_jeppeV1_df = pd.DataFrame({"radius": ttbar_radii, "fraction": fracs, "weights": jeppe_rw.tolist()})
dataFrames["ttbar_jeppeV1_df"] = {"process": "TTbar", "df": ttbar_jeppeV1_df, "xmd":jeppe_xmd_ttbarV1,"title": r"$t\bar{t}$ Andersen V1 $et$ $al.$", "color": colors[5], "marker": markers[5], "line": '--'}


#### Newer Jeppe RW TTbar
ttbar_radii = np.logspace(np.log10(125),3,50)
jeppe_rwV2 = np.load(rjainDir + "/jeppe_new_v2/ttbar_jeppe_reweights.npy")*14456.19
fracs = get_fracs(ttbar_radii, jeppe_rwV2, weights_orig_ttbar)
ttbar_jeppeV2_df = pd.DataFrame({"radius": ttbar_radii, "fraction": fracs, "weights": jeppe_rwV2.tolist()})
dataFrames["ttbar_jeppeV2_df"] = {"process": "TTbar", "df": ttbar_jeppeV2_df, "xmd":jeppe_xmd_ttbarV2,"title": r"$t\bar{t}$ Andersen V2 $et$ $al.$", "color": colors[6], "marker": markers[6], "line": '--'}

#### Newer Jeppe RW TTbar
ttbar_radii = np.logspace(2,3,50)
jeppe_rwV3 = np.load(rjainDir + "/jeppe_new_v3/ttbar_jeppe_reweights.npy")*14456.19
fracs = get_fracs(ttbar_radii, jeppe_rwV3, weights_orig_ttbar)
ttbar_jeppeV3_df = pd.DataFrame({"radius": ttbar_radii, "fraction": fracs, "weights": jeppe_rwV3.tolist()})
dataFrames["ttbar_jeppeV3_df"] = {"process": "TTbar", "df": ttbar_jeppeV3_df, "xmd":jeppe_xmd_ttbarV3,"title": r"$t\bar{t}$ Andersen $et$ $al.$", "color": colors[7], "marker": markers[7], "line": '--'}