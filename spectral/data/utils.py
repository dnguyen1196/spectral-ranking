import numpy as np
import sys

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


def from_csv(file_name, multinomial_data=False):
    def extract_line_pairwise_comparison(data):
        group = data[:-1]-1 # Reduce to 0-based indexing
        choice = data[-1]-1
        return (group, choice)

    def extract_line_multinomial_choice(data):
        items      = data[1:]
        participants = np.where(items == 1)[0]
        winner = data[0]-1 # Since it's 1-indexing
        return (participants, winner)

    def load():
        dataset = []
        items = set()
        with open(file_name, "r") as f:
            for line in f:
                data = np.array([int(x) for x in line.rstrip().split(",")])
                sample = None

                if multinomial_data:
                    sample = extract_line_multinomial_choice(data)
                else:
                    sample = extract_line_pairwise_comparison(data)
                
                for item in sample[0]:
                    items.add(item)

                dataset.append(sample)
        
        return dataset, len(items)

    return load