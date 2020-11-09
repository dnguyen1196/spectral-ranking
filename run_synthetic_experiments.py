import argparse

arg_parser = argparse.ArgumentParser("Run synthetic experiments")

arg_parser.add_argument("--n", type=int, default=50, help="Number of items")
arg_parser.add_argument("--k", type=int, default=2, help="Size of choice groups")
arg_parser.add_argument("--N", type=int, default=100, help="Number of choice groups")
arg_parser.add_argument("--top", type=str, choices=["random", "star", "chain"], default="random", help="Topology of choice graph")
arg_parser.add_argument("--n_trials", type=int, default=50, help="Number of trials")
arg_parser.add_argument("--reg_l", type=float, default=0.0, help="Regularization parameter")
arg_parser.add_argument("--output_folder", type=str, default="synthetic_exp_output", help="Output folder for synthetic experiments")
arg_parser.add_argument('--eps', nargs="+", type=float, default=[0.5, 0.75, 1, 1.5, 3], help="Epsilon values")



args = arg_parser.parse_args()
n = args.n
N = args.N
k = args.k
topology = args.top
n_trials = args.n_trials
reg_l = args.reg_l
output_folder = args.output_folder
eps = args.eps

import spectral
import numpy as np
from spectral.regularized_spectral_rank import RegularizedSpectralRank
import time
import os
import torch

L0 = 1000
n_exp = 20

# If the output folder does not exist, make it
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)


all_L_curves = []

for l in range(n_exp): # For increasing L (number of users)
    L = L0*(l+1)**2

    all_error_curves = []

    for i in range(n_trials): # For each trial

        # Generate new data
        seed = None
        np.random.seed(seed)
        scores = np.random.rand(n)
        scores_true = scores/scores.sum()
        print("w* = ", scores_true)
        synthetic_data = spectral.data.utils.generate_data_btl_by_user_with_special_topology(L=L, weights=scores_true, k=k, N=N, top=topology)

        # Run experiments
        start = time.time()
        experiment = spectral.experiment.PrivacyCurveExperiment(RegularizedSpectralRank, synthetic_data, scores_true)
        error_curve = experiment.run(eps, "rappor", seed=None)
        end = time.time()

        # Save to file
        all_error_curves.append(error_curve)
        print(error_curve["metrics"]["scores_l1"])
        print(error_curve["metrics"]["nll"])
        print(f"Trial {i} took {end - start}")

    torch.save(all_error_curves, os.path.join(output_folder, f"{topology}_L={L}_n={n}_N={N}.pkl"))
    all_L_curves.append(all_error_curves)
    
torch.save(all_L_curves, os.path.join(output_folder, f"{topology}_all_Ls_n={n}_N={N}.pkl"))