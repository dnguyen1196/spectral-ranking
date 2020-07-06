import numpy as np


def randomized_response(responses, epsilon):
    perturbed_responses = []

    for items, choice in responses:
        assert(choice in items)
        k = len(items)
        other_choice = [c for c in items if c != choice]

        if np.random.rand() > np.exp(epsilon)/(np.exp(epsilon)+k-1):
            choice = np.random.choice(other_choice)

        perturbed_responses.append((items, choice))

    return perturbed_responses
