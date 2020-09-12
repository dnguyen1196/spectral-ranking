import numpy as np
from spectral.metrics import *
from spectral.privacy import randomized_response
import collections


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
                metrics=[ranks_kendall_tau, ranks_discounted_cummulative_gain, scores_l1],
                init_param={}):

        self.estimator = estimator
        self.data = data
        self.true_scores = np.array(scores)
        self.metrics = metrics
        self.init_param = init_param
        self.true_ranks = self.get_true_ranks(scores)

    def get_true_ranks(self, scores):
        return np.array(np.flip(np.argsort(scores)))

    def run(self, epsilons=np.linspace(0, 3, 20)):
        """
        """
        error_curve = {}
        error_curve["epsilon_vals"] = epsilons
        error_curve["metrics"] = collections.defaultdict(list)
        negative_loglik = []
        for eps in epsilons:
            # Get noisy data
            noisy_data = randomized_response(self.data, eps)
            # Initialize the ranking algorithm
            estimator = self.estimator(epsilon=eps, **self.init_param)
            r_hat = estimator.fit_and_rank(noisy_data)
            nll   = negative_lik_mnl(estimator.scores, self.data)
            print(estimator.scores)
            negative_loglik.append(nll)
            for metric in self.metrics:
                # If rank metrics
                if metric.__name__.startswith("ranks"):
                    error_curve["metrics"][metric.__name__].append(metric(self.true_ranks, r_hat))
                # If score metrics
                else:
                    w_hat = estimator.get_scores()
                    error_curve["metrics"][metric.__name__].append(metric(self.true_scores, w_hat))
        
        error_curve["metrics"]["nll"] = np.array(negative_loglik)
        return error_curve


