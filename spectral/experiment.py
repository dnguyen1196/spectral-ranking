import numpy as np
from spectral.metrics import *
from spectral.privacy import randomized_response
import collections


"""
- scores vs rank 
- Rank 0 should have the highest score?

- Some metrics work with scores and some with ranks. Have to somehow separate between
these two types of metrics
"""

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
    def __init__(self, estimator, data, scores, ranks,
            metrics=[ranks_kendall_tau, ranks_discounted_cummulative_gain]):
        self.estimator = estimator
        self.data = data
        self.true_scores = scores
        self.metrics = metrics
        # TODO: compute true_ranks

    def run(self, epsilons=np.logspace(-3, 1, 10)):
        error_curve = {}
        error_curve["epsilon_vals"] = epsilons
        error_curve["metrics"] = collections.defaultdict(list)

        for eps in epsilons:
            noisy_data = randomized_response(self.data, eps)
            estimator = self.estimator()
            r_hat = estimator.fit_and_rank(noisy_data)
            
            for metric in self.metrics:
                if metric.__name__.startswith("ranks"):
                    error_curve["metrics"][metric.__name__].append(metric(self.true_ranks, r_hat))
                else:
                    w_hat = estimator.get_scores()
                    error_curve["metrics"][metric.__name__].append(metric(self.true_scores, w_hat))

        return error_curve


