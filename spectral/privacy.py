import numpy as np
import collections

def randomized_response(responses, epsilon, seed=2666):
    # The issue is that this step gets repeated all the times but we're not doing
    # any savings here, would that help
    perturbed_responses = []
    np.random.seed(seed)
    for items, choice in responses:
        assert(choice in items)
        k = len(items)
        other_choice = [c for c in items if c != choice]

        winner = choice
        if epsilon == np.inf:
            winner = choice
        elif np.random.rand() > np.exp(epsilon)/(np.exp(epsilon)+k-1):
            winner = np.random.choice(other_choice)

        perturbed_responses.append((items, winner))

    return perturbed_responses

def randomized_response_by_users(user_data, epsilon, seed=2666):
    responses = []
    for user_choices in user_data:
        responses.append(randomized_response(user_choices, epsilon, seed))
    return responses

def randomize_data_fast(data_by_choice_group, eps, seed=2666):
    responses = collections.defaultdict(list) # group_choice -> list of choices

    for (group, choices) in data_by_choice_group.items():
        if eps == np.inf:
            responses[group] = choices
        else:
            k = len(group)
            cutoff = np.exp(eps)/(np.exp(eps)+k-1)
            other_items = collections.defaultdict(list)
            for i in group:
                others = [j for j in group if j != i]
                other_items[i] = others

            L = len(choices)
            k = len(group)
            # Pre-flip L coins
            coin_flips = np.random.rand(L)

            # Pre-pick replacements (one of k-1)
            alternatives = np.random.choice(k-1, (L,))

            # Check for each choices
            y_rr = np.copy(choices)
            flipped_indices = np.argwhere(coin_flips > cutoff).flatten()

            all_other_choices = [other_items[y] for y in choices]
            all_alternatives  = [all_other_choices[ind][alternatives[ind]] for ind in flipped_indices]

            y_rr[flipped_indices] = np.array(all_alternatives)

            responses[group] = y_rr

    return responses

def rappor(responses, epsilon, seed=2666):
    perturbed_responses = []

    for items, choice in responses:
        y = np.zeros((len(items,)))

        if epsilon == np.inf: # If no privacy imposed, it's the same one hot vector
            y[choice] = 1
        else:
            for i, item in enumerate(items):
                if item == choice:
                    if np.random.rand() < (np.exp(epsilon/2)+1)/(np.exp(epsilon/2)-1):
                        y[i] = 1
                else:
                    if np.random.rand() < 1./(np.exp(epsilon/2)+1):
                        y[i] = 1
        
        perturbed_responses.append((items, y))
    return perturbed_responses

def rappor_by_user(user_data, epsilon, seed=2666):
    responses = []
    for user_choices in user_data:
        responses.append(rappor(user_choices, epsilon, seed))
    return responses

def rappor_fast(data_by_choice_group, eps, seed=2666):
    responses = collections.defaultdict(list) # group_choice -> list of ys vector

    for (group, choices) in data_by_choice_group.items():
        if eps == np.inf:
            # Convert choices to 1-hot
            L = len(choices)
            choice_ind = dict([(y, i) for i, y in enumerate(group)])
            k = len(group)
            y_onehot = np.zeros((L, k))
            for i, y in enumerate(choices):
                y_onehot[i][choice_ind[y]] = 1
            responses[group] = y_onehot

        else:
            cutoff = np.exp(eps/2)/(np.exp(eps/2)+1)
            L = len(choices)
            choice_ind = dict([(y, i) for i, y in enumerate(group)])
            k = len(group)
            y_rappor = np.zeros((L, k))
            coinflips = np.random.rand(L, k)
            y_rappor[np.argwhere(coinflips > cutoff)] = 1

            coinflips = np.random.rand(L)
            for i in range(L):
                y = choices[i]
                if coinflips[i] < cutoff:
                    y_rappor[i, choice_ind[y]] = 1
                else:
                    y_rappor[i, choice_ind[y]] = 0

            responses[group] = y_rappor

    return responses

