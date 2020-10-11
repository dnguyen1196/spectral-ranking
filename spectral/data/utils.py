import numpy as np
import sys
from math import factorial
from itertools import combinations
import operator as op
from functools import reduce
import collections

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

# TODO: re-code this function

def generate_data_MNL(N, L, weights, k, seed=2666):
    """ Generate choice data according to the MNL
    model

    Arg:
        N: number of unique choice groups
        L: number of customers
        weights: scores of items
        k: comparison group size

    Returns:
        list[(group, choice)]
    """
    dataset = []
    n = len(weights)

    # First sample for N unique k choice sets
    # from the universe of n items
    assert (N <= ncr(n, k))
    np.random.seed(seed)
    all_groups = list(combinations(np.arange(n), k))
    np.random.shuffle(all_groups)
    unique_groups = all_groups[:N]

    dataset = []
    for group in unique_groups:
        for i in range(L): # For each user
            probs = np.array([weights[i] for i in group])
            probs = probs / np.sum(probs) # Normalize
            winner = np.random.choice(group, 1, p=probs)[0]
            dataset.append((list(group), winner))

    np.random.shuffle(dataset)
    return dataset

def generate_data_MNL_by_user(N, L, weights, k, seed=2666):
    """ Generate choice data according to the MNL
    model but returns the data separated into different users

    Arg:
        N: number of unique choice groups
        L: number of customers
        weights: scores of items
        k: comparison group size

    Returns:
        list[user_choice]
            where user_choice is list[(group, winner)]
    """
    dataset = []
    n = len(weights)

    # First sample for N unique k choice sets
    # from the universe of n items
    assert (N <= ncr(n, k))
    np.random.seed(seed)
    all_groups = list(combinations(np.arange(n), k))
    np.random.shuffle(all_groups)
    unique_groups = all_groups[:N]

    dataset = []
    for i in range(L): # For each user
        user_choice = []
        for group in unique_groups:
            probs = np.array([weights[i] for i in group])
            probs = probs / np.sum(probs) # Normalize
            winner = np.random.choice(group, 1, p=probs)[0]
            user_choice.append((list(group), winner))
        dataset.append(user_choice)

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


def group_by_choice_sets(dataset):
    grouped_data = collections.defaultdict(list)

    for choice_group, winner in dataset:
        grouped_data[tuple(choice_group)].append(winner)

    return grouped_data

def group_by_user(dataset):
    """

    all_data:
        list[user_choice]
            where
            user_choice: []

    """
    data_by_choiceset = group_by_choice_sets(dataset)

    choice_groups = list(data_by_choiceset.keys())
    num_responses = [len(g) for _, g in data_by_choiceset.items()]
    L             = max(num_responses)

    all_data = []
    for i in range(L):
        user_choice = []
        for choice_set in choice_groups:
            winning_items = data_by_choiceset[choice_set]
            if i < len(winning_items):            
                user_choice.append((list(choice_set), winning_items[i]))
        all_data.append(user_choice)

    return all_data, choice_groups