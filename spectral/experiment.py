import numpy as np
from spectral.metrics import *
from spectral.privacy import randomized_response, randomized_response_by_users,\
    rappor_by_user, randomize_data_fast, rappor_fast
from spectral.data.utils import aggregate_by_choice_groups
import collections
import time

class LearningCurveExperiment():
    def __init__(self, estimator, data, scores=None, ranks=None,
            metrics=[ranks_kendall_tau, ranks_discounted_cummulative_gain]):
        self.estimator = estimator
        self.data = data
        self.N    = len(data)

        assert((scores is None or ranks is None) and not (scores is None and ranks is None))

        if scores is None:
            self.ranks = ranks
            self.scores = None
        else:
            self.scores = scores
            self.ranks = np.argsort(scores)

    def run(self, data_curve=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        """
        
        """
        error_curve = {}
        error_curve["data_volume"] = data_curve
        error_curve["metrics"] = collections.defaultdict(list)
        for vol in data_curve:
            # Get a small amount of data
            train_data = data[:int(vol * self.N)]
            estimator = self.estimator(self.n_items)
            r_hat = estimator.fit_and_rank(train_data)

            # Record metrics
            for metric in self.metrics:
                if metric.__name__.startswith("ranks"):
                    error_curve["metrics"][metric.__name__].append(metric(self.ranking, r_hat))
                else:
                    w_hat = estimator.get_scores()
                    error_curve["metrics"][metric.__name__].append(metric(self.scores, w_hat))

        return error_curve

class PrivacyCurveExperiment():
    def __init__(self, estimator, data, scores,
                metrics=[
                    scores_l1,
                    nll
                ],
                data_by_user=False,
                init_param={}):

        self.estimator = estimator
        self.data = data
        self.metrics = metrics
        self.init_param = init_param
        if scores is not None:
            self.true_scores = np.array(scores)
            self.true_ranks = self.get_true_ranks(scores)

    def get_true_ranks(self, scores):
        return np.array(np.flip(np.argsort(scores)))

    def run(self, epsilons=np.linspace(0, 3, 20), mechanism="rr", seed=2666):
        """
        """
        error_curve = {}
        error_curve["epsilon_vals"] = epsilons
        error_curve["metrics"] = collections.defaultdict(list)
        error_curve["learned_scores"] = []
        negative_loglik = []
        self.learned_scores = []

        true_data_by_choice_groups = aggregate_by_choice_groups(self.data)

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
            end = time.time()

        return error_curve


