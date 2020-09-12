import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import dcg_score
from scipy.stats import spearmanr

def ranks_kendall_tau(r_hat, r):
    """ Compute the Kendall Tau distance bewteen two
    rankings
    """
    assert(r_hat.shape == r.shape)
    return kendalltau(r_hat, r).correlation

def ranks_discounted_cummulative_gain(r_hat, r):
    """ Compute the discounted cummulative gain

    Assuming that the items are labeled 0, ..., n_items
    """
    score = {}
    n_items   = len(r)
    for i, item in enumerate(r):
        score[item] = n_items - i

    dcg = 0.
    for i, item in enumerate(r_hat):
        dcg += score[item]/ np.log2(i+2)

    return dcg

def ranks_spearman_rho(r_hat, r):
    """ Compute the spearman rank coefficient of
    two rankings
    """
    assert(r_hat.shape == r.shape)
    n_items = len(r_hat)
    x = np.zeros((n_items,))
    y = np.zeros((n_items,))

    for i, item in enumerate(r):
        x[item] = i+1
    for i, item in enumerate(r_hat):
        y[item] = i+1

    return spearmanr(x, y).correlation

def scores_l1(w, w_hat):
    """ L1 distance between two weight vectors
    """
    assert(w.shape == w_hat.shape)
    return np.sum(np.abs(w-w_hat))

def negative_lik_mnl(w_hat, data):
    """ Compute the negative log likelihood of the data
    under the MNL model
    """
    nll = 0.
    for (group, choice) in data:
        score_group = np.array([w_hat[i] for i in group])
        p_choice_data = w_hat[choice]/score_group.sum()
        if p_choice_data == 0: # Avoid overflown error
            continue
        nll -= np.log(p_choice_data)
    return nll