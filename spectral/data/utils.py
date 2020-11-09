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

    items = set()
    for g in unique_groups:
        for item in g:
            items.add(item)
    untouched = [i for i in range(n) if i not in items]
    for i in untouched:
        g = list(np.random.choice(n, k-1, False)) + [i]
        unique_groups.append(np.array(g))
    

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

def generate_tree_graph(n, k=2, width=1, seed=2666):
    """
    n: Number of items
    """
    choice_groups = []
    unvisited     = set(list(range(n)))
    visited       = set()
    assert(width < k)

    np.random.seed(seed)
    root_group = np.random.choice(list(range(n)), k, False)

    for i in root_group:
        unvisited.remove(i)
        visited.add(i)
    choice_groups = [root_group]
    frontier = [root_group]

    while len(frontier) > 0 and len(unvisited) > 0:
        par = frontier.pop(0)
        # Pick some item to extend a new comparison group
        next_pars = np.random.choice(par, width, False)
        for p in next_pars:
            if len(unvisited) < k:
                next_group = list(unvisited) \
                    + list(np.random.choice(list(visited), k-len(unvisited), False))
                for i in next_group:
                    if i in unvisited:
                        unvisited.remove(i)
                    visited.add(i)
                choice_groups.append(next_group)
                break

            # Pick from the remaining unvisited
            next_group = list(np.random.choice(list(unvisited), k-1, False))
            next_group.append(p)

            choice_groups.append(next_group)
            frontier.append(next_group)
            for i in next_group:
                if i in unvisited:
                    unvisited.remove(i)
                visited.add(i)
    return choice_groups

def generate_chain_graph(n, k=2, seed=2666):
    """
    
    """
    assert (k >= 2)
    choice_groups = []

    np.random.seed(seed)
    chain = list(range(n))
    np.random.shuffle(chain)

    i = 0
    while i < n:
        if i + k > n:
            choice_groups.append(chain[-k:])
            break

        group = chain[i: i+k]
        choice_groups.append(group)
        i = i+k

    return choice_groups


def generate_cycle_graph(n, k=2, seed=2666):
    """
    
    """
    choice_groups = []

    np.random.seed(seed)
    chain = list(range(n))
    np.random.shuffle(chain)

    i = 0
    while i < n:
        if i + k > n:
            group = chain[-k:]
            choice_groups.append(group)
            break

        group = chain[i: i+k]
        choice_groups.append(group)
        i = i+k

    group = [chain[-1]]
    group.extend(chain[:k-1])
    if (np.all(choice_groups[-1] == np.array(group))):
        return choice_groups

    choice_groups.append(group)
    return choice_groups


def generate_star_graph(n, k=2, seed=2666):
    """
    Generate a graph where there is only one center and the remaining
    vertiices are connected only to the center
    """
    choice_groups = []
    # assert ((k == 2) or (n % (k-1) == 1)) # Simple fix to make sure we get equal group size
    np.random.seed(seed)
    center = np.random.choice(n)

    other = set([item for item in range(n) if item != center])

    while (len(other) > 0):
        if (len(other)) < k-1:
            group_mem = list(other)
            group_mem.append(center)

            group = group_mem
        else:
            group_mem = np.random.choice(list(other), k-1, False)
            group = list(group_mem)
            group.append(center)
            
        choice_groups.append(group)
        for i in group:
            if i in other:
                other.remove(i)

    return choice_groups


def generate_random_graph(n, N, k=5, seed=2666):
    # First sample for N unique k choice sets
    # from the universe of n items
    assert (N <= ncr(n, k))
    np.random.seed(seed)
    all_groups = list(combinations(np.arange(n), k))
    np.random.shuffle(all_groups)
    unique_groups = all_groups[:N]

    items = set()
    for g in unique_groups:
        for item in g:
            items.add(item)
    untouched = [i for i in range(n) if i not in items]
    for i in untouched:
        other = set(list(range(n)))
        other.remove(i)
        other_mem = np.random.choice(list(other), k-1, False)
        g = list(other_mem) + [i]
        unique_groups.append(g)

    return unique_groups


def generate_data_from_choice_groups(choice_groups, L, weights, seed=2666):
    dataset = []

    all_data = []
    np.random.seed(seed)
    for group in choice_groups:
        probs = np.array([weights[i] for i in group])
        probs = probs / np.sum(probs) # Normalize

        # Sample L times with specified probabilities
        samples = np.random.choice(group, p=probs, size=(L,))
        all_data.append(samples)

    for l in range(L): # For each user
        user_choice = []
        for a, sample in enumerate(all_data):
            user_choice.append((choice_groups[a], sample[l]))
        dataset.append(user_choice)

    return dataset


def generate_choice_groups_by_topology(n, k=2, N=100, top="tree", seed=2666):
    if top=="random":
        choice_groups = generate_random_graph(n, N, k, seed)
    elif top == "tree":
        choice_groups = generate_tree_graph(n, k, k-1, seed)
    elif top == "chain":
        choice_groups = generate_chain_graph(n, k, seed)
    elif top == "cycle":
        choice_groups = generate_cycle_graph(n, k, seed)
    else:
        choice_groups = generate_star_graph(n, k, seed)
    return choice_groups


def generate_data_btl_by_user_with_special_topology(L, weights, k=2, N=100, top="tree", seed=2666):

    n = len(weights)
    choice_groups = generate_choice_groups_by_topology(n, k, N, top, seed)

    dataset = []

    all_data = []
    np.random.seed(seed)
    for group in choice_groups:
        probs = np.array([weights[i] for i in group])
        probs = probs / np.sum(probs) # Normalize

        # Sample L times with specified probabilities
        samples = np.random.choice(group, p=probs, size=(L,))
        all_data.append(samples)

    for l in range(L): # For each user
        user_choice = []
        for a, sample in enumerate(all_data):
            user_choice.append((choice_groups[a], sample[l]))
        dataset.append(user_choice)

    return dataset


def aggregate_by_choice_groups(users_data):
    """

    data: list[N]
        data[a] = list[La]
    """
    data_by_group = collections.defaultdict(list)
    for user_data in users_data:
        for (group, y) in user_data:
            data_by_group[tuple(group)].append(y)

    return data_by_group