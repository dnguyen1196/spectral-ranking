import argparse
import numpy as np

arg_parser = argparse.ArgumentParser("Run REAL experiments")

arg_parser.add_argument("--data", type=str, help="Data set", choices=["youtube", "sfshop", "sfwork"])
arg_parser.add_argument("--n_trials", type=int, default=50, help="Number of trials")
arg_parser.add_argument("--reg_l", type=float, default=0.0, help="Regularization parameter")
arg_parser.add_argument("--output_folder", type=str, default="real_exp_output", help="Output folder for synthetic experiments")
arg_parser.add_argument('--eps', nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75, 1, 1.5, 3, np.inf], help="Epsilon values")
arg_parser.add_argument("--method", type=str, default="rappor", help="Privatization method", choices=["rappor", "rr"])

args = arg_parser.parse_args()

n_trials = args.n_trials
reg_l = args.reg_l
output_folder = args.output_folder
eps = args.eps
data_set = args.data
method = args.method

import spectral
from spectral.regularized_spectral_rank import RegularizedSpectralRank
import time
import os
import torch
from spectral.metrics import *

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:min(i + n, len(lst))]
        
def split(lst, L):
    n = int(len(lst)/L)+1
    return chunks(lst, n)

# If the output folder does not exist, make it
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
if data_set == "youtube":
    user_data, _ = spectral.data.youtube()
    L = 10000
    data = [dat for dat in split(user_data, L)]
elif data_set == "sfshop":
    user_data, _ = spectral.data.sfshop()
    data, _ = spectral.data.utils.group_by_user(user_data)
else:
    user_data, _ = spectral.data.sfwork()
    data, _ = spectral.data.utils.group_by_user(user_data)

all_error_curves = []
print("Loaded data, working....")
for i in range(n_trials): # For each trial
    # Run experiments
    start = time.time()
    experiment = spectral.experiment.PrivacyCurveExperiment(RegularizedSpectralRank, data, None, metrics=[nll])
    error_curve = experiment.run(eps, method, seed=None)
    end = time.time()

    # Save to file
    all_error_curves.append(error_curve)
    print(error_curve["metrics"]["nll"])
    print(f"Trial {i} took {end - start}")

    torch.save(all_error_curves, os.path.join(output_folder, f"{data_set}.pkl"))