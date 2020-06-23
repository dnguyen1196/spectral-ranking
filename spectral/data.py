import numpy as np
import sys

# Generate synthetic data

available_models = {
    "BTL" : generate_data_BTL,
    "MNL" : generate_data_MNL,
    "Mallows" : generate_data_mallows,
}


def generate_data(N, model, *args, **kwargs):
    if not (model in available_models):
        print("Model not available")
        sys.exit(1)
    return available_models[model](N, *args, **kwargs)


def generate_data_BTL(N, weights):
    """ Generate pairwise comparison data according to the 
    BTL model

    Arg:
        N
        weights
    Returns:
        comparisons
    """

    return

def generate_data_MNL(N, weights, k):
    """ Generate choice data according to the MNL
    model

    Arg:


    Returns:

    """


    return

def generate_data_mallows(N, mean, scale):

    return