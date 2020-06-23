import numpy as np
from spectral.metrics import *

class LearningCurveExperiment():
    def __init__(self, estimator, X, y, metrics=[kendall_tau, discounted_cummulative_gain]):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.metrics = metrics

    def run(self):
        """ Partition the training data into growing size
        """

        return
