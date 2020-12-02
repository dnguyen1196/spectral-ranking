import numpy as np
import scipy
from spectral.metrics import *
from spectral.privacy import randomized_response, randomized_response_by_users,\
    rappor_by_user, randomize_data_fast, rappor_fast
from spectral.data.utils import aggregate_by_choice_groups
import collections
import time
import itertools
from scipy.stats._stats import _kendall_dis

def kendall_tau_distance(lst_a, lst_b):
    order_a = list(lst_a)
    order_b = list(lst_b)
    pairs = itertools.combinations(range(0, len(order_a)), 2)
    distance = 0
    for x, y in pairs:
        a = order_a.index(x) - order_a.index(y)
        b = order_b.index(x) - order_b.index(y)
        if a * b < 0:
            distance += 1
    return distance

def accuracy_heldout_data(r_hat, heldout_data_by_choice_groups, top=2):
    """Given a predicted ranking and heldout data organized by choice
    groups, evaluate the accuracy of predicting winning items using
    r_hat

    :param r_hat: [description]
    :type r_hat: [type]
    :param heldout_data_by_choice_groups: [description]
    :type heldout_data_by_choice_groups: [type]
    """
    num_correct = 0
    num_rounds = 0
    for choice_group, choices in heldout_data_by_choice_groups.items():
        
        ranked_choice_group = []
        for item in r_hat:
            if item in choice_group:
                ranked_choice_group.append(item)

        for y in choices:
            if (y in ranked_choice_group[:top]):
                num_correct += 1
        num_rounds += len(choices)

    return float(num_correct)/num_rounds


def kendall_tau(r_hat, heldout_rankings):
    """[summary]

    :param r_hat: [description]
    :type r_hat: [type]
    :param ranked_heldout_data: [description]
    :type ranked_heldout_data: [type]
    :return: [description]
    :rtype: [type]
    """
    n = len(heldout_rankings)
    avg_kt = 0.
    for ranking in heldout_rankings:
        # print(ranking, "vs", r_hat)
        tau, p_value = scipy.stats.kendalltau(r_hat, ranking)
        avg_kt += 1./n * tau
    return avg_kt


def avg_kendall_tau_distance(r_hat, heldout_rankings):
    avg_kd = 0.
    n = len(heldout_rankings)

    for ranking in heldout_rankings:
        # kd = _kendall_dis(bytearray(r_hat), bytearray(ranking))
        kd = kendall_tau_distance(r_hat, ranking)
        avg_kd += 1./n * kd
    return avg_kd


def spearman_rho(r_hat, heldout_rankings):
    avg_rho = 0.
    n = len(heldout_rankings)

    for ranking in heldout_rankings:
        rho = ranks_spearman_rho(r_hat, ranking)
        avg_rho += 1./n * rho

    return avg_rho


def discounted_cummulative_gain(r_hat, heldout_rankings):
    avg_cdg = 0.
    n = len(heldout_rankings)
    for ranking in heldout_rankings:
        cdg = ranks_discounted_cummulative_gain(r_hat, ranking)
        avg_cdg += 1./n * cdg
    return avg_cdg


class PrivacyCurveExperiment():
    def __init__(self, estimator, data, scores,
                heldout_data=None,
                heldout_rankings=None,
                metrics=[
                    scores_l1,
                    nll
                ],
                init_param={}):

        self.estimator = estimator
        self.data = data
        self.heldout = heldout_data
        self.heldout_rankings = heldout_rankings
        self.metrics = metrics
        self.init_param = init_param
        if scores is not None:
            self.true_scores = np.array(scores)
            self.true_ranks = self.get_true_ranks(scores)

    def get_true_ranks(self, scores):
        return np.array(np.flip(np.argsort(scores)))

    def run(self, epsilons=np.linspace(0, 3, 20), mechanism="rr", seed=2666, top_k_pred=1):
        """
        """
        error_curve = {}
        error_curve["epsilon_vals"] = epsilons
        error_curve["metrics"] = collections.defaultdict(list)
        error_curve["learned_scores"] = []
        negative_loglik = []
        self.learned_scores = []

        true_data_by_choice_groups = aggregate_by_choice_groups(self.data)

        heldout_data_by_choice_groups = None
        if self.heldout is not None:
            heldout_data_by_choice_groups = aggregate_by_choice_groups(self.heldout)

        # Delete the data to save some space
        del(self.data)

        for eps in epsilons:
            start = time.time()
            if mechanism == "rr":
                noisy_data_by_group = randomize_data_fast(true_data_by_choice_groups, eps, seed=seed)
            else:
                noisy_data_by_group = rappor_fast(true_data_by_choice_groups, eps, seed=seed)
                
            end = time.time()

            # Initialize the ranking algorithm
            start = time.time()
            estimator = self.estimator(epsilon=eps, mechanism=mechanism, **self.init_param)
            r_hat = estimator.fit_and_rank(noisy_data_by_group)
            end = time.time()

            # Save the learned scores
            self.learned_scores.append(estimator.get_scores())
            error_curve["learned_scores"].append(estimator.get_scores())

            # Evaluate NLL on ACTUAL private data
            start = time.time()

            for metric in self.metrics:
                # If rank metrics
                if metric.__name__.startswith("ranks"):
                    error_curve["metrics"][metric.__name__].append(metric(self.true_ranks, r_hat))

                # If negative log likelihood
                elif metric.__name__.startswith("nll"):
                    error_curve["metrics"][metric.__name__].append(
                        nll(estimator.scores, true_data_by_choice_groups))
                else:
                    # If score metrics
                    w_hat = estimator.get_scores()
                    error_curve["metrics"][metric.__name__].append(metric(self.true_scores, w_hat))

            if heldout_data_by_choice_groups is not None:
                # Do prediction on heldout dataset
                acc = accuracy_heldout_data(r_hat, heldout_data_by_choice_groups, top=top_k_pred)
                error_curve["metrics"]["accuracy"].append(acc)

            if self.heldout_rankings is not None:
                avg_kt = kendall_tau(r_hat, self.heldout_rankings)
                error_curve["metrics"]["kendall_tau"].append(avg_kt)
                error_curve["metrics"]["spearman_rho"].append(spearman_rho(r_hat, self.heldout_rankings))
                error_curve["metrics"]["dcg"].append(discounted_cummulative_gain(r_hat, self.heldout_rankings))
                error_curve["metrics"]["kendall_tau_distance"].append(avg_kendall_tau_distance(r_hat, self.heldout_rankings))

            end = time.time()

        return error_curve


