import numpy as np
import sys

# Generate synthetic data

available_models = {
    "MNL" : generate_data_MNL,
    "Mallows" : generate_data_mallows,
}


def generate_data(N, model, *args, **kwargs):
    if not (model in available_models):
        print("Model not available")
        sys.exit(1)
    return available_models[model](N, *args, **kwargs)

def generate_data_MNL(N, weights, k):
    """ Generate choice data according to the MNL
    model

    Arg:
        N: number of comparisons
        weights: scores of items
        k: comparison group size

    Returns:
        list[(group, choice)]
    """
    dataset = []
    n = len(weights)

    for i in range(N):
        group = np.random.choice(n, (k,), replace=False)
        probs = [weights[i] for i in group]
        winner = np.random.choice(group, 1, p=probs)
        dataset.append((probs, winner))

    return dataset

def generate_data_mallows(N, mean, scale):
    
    return


def from_csv(file_name):
    def load():
        dataset = []
        items = set()
        with open(file_name, "r") as f:
            for line in f:
                data = [int(x) for x in line.rstrip().split(",")]
                group = data[:-1]
                items.add(group)
                choice = data[-1] # Is choice item always the last?
                dataset.append((group, choice))
        
        return dataset, len(items)
    return load