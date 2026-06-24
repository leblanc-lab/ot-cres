import numpy as np
import os
import ot
import warnings
import pandas as pd

import argparse
import numpy as np

def get_fracs(radii, weights, weights_orig):
    fracs = []
    for R, weight in zip(radii, weights):
        frac = (1-np.sum(weight<0)/np.sum(weights_orig<0))
        fracs.append(round(frac, 2))
    return fracs

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
        
def parse_args():
    parser = argparse.ArgumentParser(description="Compute XMD distances for a set of reweights.")

    # Input files
    parser.add_argument("--weights_orig", required=False, help="Path to original weights (.npy)",)
    parser.add_argument("--distmatrix", required=False, help="Path to distance matrix (.npy)", )
    parser.add_argument("--radii", type=float, nargs="+", required=False, 
                        help="List of radii (space separated)", )
    parser.add_argument("--ttbar", action='store_true', help="Use ttbar files and reweights")
    parser.add_argument("--jeppe", action='store_true', help="Use jeppe files and reweights")
    parser.add_argument("--test", action='store_true', help="Use only 10k files")
    # parser.add_argument("--reweights", help="Path to stacked reweights .npy")
    # parser.add_argument("--reweight_prefix", help="Prefix for per-radius files")
    # parser.add_argument("--reweight_suffix", default="gev_bigR.npy")
    parser.add_argument("--stage", type=int, default=2, 
                        help="Stage of generation to evaluate: 0 - hard process, 1 - parton shower, 2 - hadronization (default)", )
    # Output
    parser.add_argument(
        "--output",
        default="xmd_distances.npy",
        help="Output filename (without extension optional)",
    )

    # Compute params
    parser.add_argument( "--threads", type=int, default=None, help="Number of threads for EMD")
    parser.add_argument("--itermax", type=int, default=10000000,help="Maximum iterations for EMD" )
    parser.add_argument("--const", type=float, default=None,help="Optional manual constant shift")

    return parser.parse_args()
    

def load_reweights_from_template(
    radii,
    input_prefix,
    input_suffix="gev_bigR.npy",
    weights_orig=None,
    points=None,
):
    weights = []

    for R in radii:
        str_R = str(R).replace(".", "p")
        path = f"{input_prefix}{str_R}{input_suffix}"

        w = np.load(path)

        if points is not None:
            w = fix_weight_length(points, w, weights_orig)

        weights.append(w)

    return np.stack(weights)  # shape (N_radii, N_events)


class XMDComputer:
    def __init__(
        self,
        original_weights,
        distmatrix,
        normalize_distmatrix=True,
        num_iter_max=10_000_000,
        num_threads=None,
        const=None,
    ):
        self.original_weights = np.asarray(original_weights, dtype=float)
        self.distmatrix = np.asarray(distmatrix, dtype=float)
        if normalize_distmatrix:
            max_dist = np.max(self.distmatrix)
            if max_dist <= 0:
                raise ValueError("Distance matrix maximum must be positive.")
            self.distmatrix = self.distmatrix / max_dist
        self.num_iter_max = num_iter_max
        self.num_threads = num_threads or os.cpu_count()

        # If not provided, const is determined later using original + all reweights
        self.const = const
    @classmethod
    def from_files(
        cls,
        original_weight_file,
        distmatrix_file,
        **kwargs,
    ):
        original_weights = np.load(original_weight_file)
        distmatrix = np.load(distmatrix_file)

        return cls(
            original_weights=original_weights,
            distmatrix=distmatrix,
            **kwargs,
        )

    def _get_constant(self, reweights):
        """
        Choose a constant large enough to make both original and reweighted
        weights non-negative.
        """
        if self.const is not None:
            return self.const
        print("orig weights min ", min(self.original_weights))
        print(reweights)
        rw_min = np.min(reweights)
        print("rw min ", rw_min)
        
        global_min = min(self.original_weights)

        return abs(global_min-0.00001) if global_min < 0 else 0.0

    def _prepare_weights(self, reweight, const):
        """
        Shift original and reweighted weights by const.
        """
        w_orig = self.original_weights + const
        w_rw = np.asarray(reweight, dtype=float) + const

        if len(w_orig) != len(w_rw):
            raise ValueError(
                f"Weight length mismatch: original has {len(w_orig)}, "
                f"reweight has {len(w_rw)}."
            )

        if np.any(w_orig < 0):
            warnings.warn(
                "Original weights still contain negative values after shifting.",
                UserWarning,
            )

        if np.any(w_rw < 0):
            warnings.warn(
                "Reweighted weights still contain negative values after shifting.",
                UserWarning,
            )

        return w_orig, w_rw

    def xmd_distance(self, w1_pos, w2_pos):  
        """
        Compute XMD between two positive weight arrays using the stored
        distance matrix.

        w1 and w2 should already be shifted to be non-negative.
        """
        #remove all events that are unaffected by reweighting
        mask0 = w1_pos != w2_pos
    
        w1_pos = w1_pos[mask0]
        w2_pos = w2_pos[mask0]
        pos_distmatrix = self.distmatrix[mask0][:, mask0]
    
        #remove all negative events that have a weight of 0 after adding constant
        mask1 = w1_pos != 0
        mask2 = w2_pos != 0
    
        w1_pos = w1_pos[mask1]
        w2_pos = w2_pos[mask2]
        pos_distmatrix = pos_distmatrix[mask1][:, mask2]
    
        print(f'Sparse distance matrix has shape: {np.shape(pos_distmatrix)}')
    
        xmd = ot.emd2( w1_pos/w1_pos.sum(), w2_pos/w2_pos.sum(), pos_distmatrix, numItermax = self.num_iter_max, numThreads = self.num_threads)
    #    pos_xmd = ot.sinkhorn2(w1_pos/w1_pos.sum(), w2_pos/w2_pos.sum(), pos_distmatrix/biggest_emd, reg = 1e-1, warn = True)
        xmd *= min(w1_pos.sum(), w2_pos.sum())
    
        print('-----------------------------------------') 
        print(f'Total distance: {xmd}')
        print('-----------------------------------------') 
        return(xmd)
    def compute_reweights(self, radii, reweights):
        """
        Compute XMD for a nested reweights array.

        Expected shape:
            reweights[i] corresponds to radii[i]
        """

        if len(radii) != len(reweights):
            raise ValueError(
                f"radii and reweights length mismatch: "
                f"{len(radii)} vs {len(reweights)}."
            )
        print(np.array(reweights))
        const = self._get_constant(np.array(reweights))
        print(f"Using constant shift: {const}")

        distances = []

        for radius, reweight in zip(radii, reweights):
            print("------------------------------------------------------------")
            print(f"Now starting: max radius = {radius}")
            print("------------------------------------------------------------")

            w_orig, w_rw = self._prepare_weights(np.array(reweight), const)
            distance = self.xmd_distance(w_rw, w_orig)
            print("XMD ", distance)
            distances.append(distance)

        return np.asarray(distances)
    def compute_and_save(self, radii, reweights, output_file):
        distances = self.compute_reweights(radii, reweights)
        np.save(output_file, distances)
        return distances

def main():
    args = parse_args()

    ##### load hadronization points
    ### test w/ 10k

    if args.ttbar:
        points = np.load('/oscar/data/mleblan6/lhay/ttbar_100k/hadronization_ttbar_points.npy')
        distmatrix = np.load("/oscar/data/mleblan6/lhay/ttbar_distmatrix/full_distmatrix_ttbar.npy")
        weights_orig = np.load("/oscar/data/mleblan6/lhay/ttbar_100k/ttbar_weight_100k.npy")
        if args.stage == 0:
            #### making the ttbar hp dataframes
            radii = [5,15,17,18.5,20,22,25,30,100]
            df = make_df(radii, weights_orig, 
                                  "/oscar/data/mleblan6/lhay/reweighted_files/ttbar_hp_100k/100k_hp_emd_reweight_")
            reweights = np.stack(df["weights"].to_numpy()/(1.455 * 10**4))
        elif args.stage == 1:
        #### make rest of ttbar dataframes

            radii = [1, 15, 18, 24, 25, 30, 50, 100]
            df = make_df(radii, weights_orig, "/oscar/data/mleblan6/lhay/reweighted_files/ttbar_ps_100k/100k_ps_emd_reweight_", "gev_ttbar.npy")
            reweights = np.stack(df["weights"].to_numpy()/(1.455 * 10**4))
        elif args.stage == 2:
            radii = [1, 18, 25, 28, 30, 50, 100]
            df = make_df(radii, weights_orig, "/oscar/data/mleblan6/lhay/reweighted_files/ttbar_had_100k/100k_had_emd_reweight_", "gev_ttbar.npy")
            reweights = np.stack(df["weights"].to_numpy()/(1.455 * 10**4))
    elif args.test:
        points = np.load("/users/lhay/NegativeWeights/hadronization_ttbar_points10k.npy")
        distmatrix = np.load("/users/lhay/NegativeWeights/distmatrix_ttbar10k.npy")
        weights_orig = np.load("/users/lhay/NegativeWeights/ttbar_weight_10k.npy")
        radii = [1, 15, 18, 24, 25, 30, 50, 100]
        df = make_df(radii, weights_orig, "/oscar/data/mleblan6/lhay/reweighted_files/10k_ttbar_ps/10k_ps_emd_reweight_", "gev_ttbar.npy")
        reweights = np.stack(df["weights"].to_numpy()/(1.455 * 10**4))
    else:
        points = np.load('/oscar/data/mleblan6/rjain/ppzjj_100k/hadronization_points.npy')
        weights_orig = np.load('/oscar/data/mleblan6/rjain/ppzjj_100k/weight_100k.npy')
        distmatrix = np.load('/oscar/data/mleblan6/rjain/xmd_groundmetrics/had_100k_distmatrix/full_distmatrix.npy')

    computer = XMDComputer(
            original_weights=weights_orig,
            distmatrix=distmatrix,
            num_threads=args.threads,
            num_iter_max=args.itermax,
            const=args.const,
        )
    print("initialized xmd computer class")

    if args.jeppe==True and args.ttbar==True:
        radii = np.logspace(np.log10(50),np.log10(500),50)
        reweights = np.load("/oscar/data/mleblan6/rjain/jeppe/ttbar_jeppe_reweights.npy")
    elif args.jeppe==True and args.ttbar==False:
        radii = np.logspace(np.log10(50),np.log10(500),50)
        reweights = np.load("/oscar/data/mleblan6/rjain/jeppe/jeppe_reweights.npy")
        
    computer.compute_and_save(
        radii=radii,
        reweights=reweights,
        output_file=args.output,
    )
    print("computed and saved file")
if __name__ == "__main__":
    main()

