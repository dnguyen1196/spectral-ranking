import numpy as np


def randomized_response(responses, epsilon):
    perturbed_responses = []

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


def randomized_response_by_users(user_data, epsilon_overall):
    responses = []
    for user_choices in user_data:
        N = len(user_choices)
        responses.append(randomized_response(user_choices, epsilon_overall/N))
    return responses
