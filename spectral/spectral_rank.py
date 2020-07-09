import numpy as np
import cvxpy as cp
import collections

class Aggregator():
    """ Aggregate the choice data to produce 
    (choice_set, statistics) tuples


    Attributes:
        epsilon: privacy parameter

        all_items: the set of all items

        group_choice: the dictionary of group -> preference frequency

        ds_array: an array storing the di's

    """
    def __init__(self, epsilon, reg_k=0):
        self.epsilon = epsilon
        self.reg_k = reg_k

    def aggregate_raw_statistics(self, data):
        """ Aggregate statistics
        """
        # Go through each comparison group but also keep track of the
        # number of times certain items win
        group_choice = {}

        # Keep all the items in the universe
        self.all_items = set()

        for group, choice in data:
            self.all_items.update(group)
        n_items = len(self.all_items)

        for group, choice in data:
            key = frozenset(group)
            self.all_items.update(key)
            if key not in group_choice:
                group_choice[key] = collections.defaultdict(int)
            group_choice[key][choice] += 1
        
        for group, choice in group_choice.items():
            # Get the m empirical estimate vector
            group_items = list(group)

            # Add regularization term
            m = np.array([choice[item] + self.reg_k for item in group_items])
            m = m/ np.sum(m)

            if self.epsilon == np.inf:
                p_hat = m
            else:
                p_hat = m * (1 + np.exp(self.epsilon))/(np.exp(self.epsilon) - 1)\
                    - 1./(np.exp(self.epsilon)+1)
                p_hat = self.project_to_probability_simplex(p_hat)

            for i, item in enumerate(group_items):
                group_choice[group][item] = p_hat[i]

        # Keep track of di
        self.ds_array  = np.zeros((n_items,))
        for group in group_choice.keys():
            for i in group:
                self.ds_array[i] += 1

        self.group_choice = group_choice

        return self.group_choice, self.all_items, self.ds_array

    def project_to_probability_simplex(self, v):
        """ Project a vector onto the probability simplex by solving the
        quadratic program:

        minimize 1/2 || x - v ||^2
        
        such that: 
                x >= 0
                x @ 1 = 1
        """
        k = len(v)
        x = cp.Variable(k)

        objective = cp.Minimize(0.5 * cp.norm2(x-v)**2)
        constraints = [cp.sum(x) == 1, x >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Remove negative numbers and normalize
        x_hat = np.maximum(x.value, 0)
        x_hat = x_hat / x_hat.sum() 
        return x_hat


class SpectralRank():
    def __init__(self, epsilon, reg_k=0, max_iters=1000, tol=1e-6):
        """

        Attributes:

        """
        self.epsilon = epsilon
        self.reg_k = reg_k
        self.aggregator = Aggregator(epsilon, reg_k)
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, data):
        """ Learn the BTL/MNL model
        """
        # Aggregate the statistics
        self.group_choice, self.all_items, self.ds_array\
            = self.aggregator.aggregate_raw_statistics(data)
        # Construct matrix P
        self.construct_P()
        # Do power iteration
        self.find_stationary_distribution()

    def fit_and_rank(self, data):
        """ 
        Return the list of items, from highest score to lowest score
        """
        self.fit(data)
        return self.get_ranks()

    def get_scores(self):
        """ Get the scores of the items
        """
        return self.scores

    def get_ranks(self):
        """
        Return the list of items (indices), 
        from highest score to lowest score
        """
        assert(hasattr(self, "scores"))
        return np.flip(np.argsort(self.scores))

    def construct_P(self):
        """

        Assumption:
            The set of items is 0-based indexing, [n]
        """
        all_items = self.all_items
        self.n = len(all_items)
        self.P = np.zeros((self.n, self.n))
        d = self.ds_array

        for (group, p_Sa) in self.group_choice.items():
            for i in group:
                for j in group:
                    self.P[i, j] += 1./d[i] * p_Sa[j]
            
    def find_stationary_distribution(self):
        """ Compute the stationary distribution and the scores
        """
        # Start with a uniform distribution
        self.pi = np.ones((self.n,)) / self.n

        for i in range(self.max_iters):
            next_pi = np.matmul(self.P.T, self.pi)
            if np.sum((next_pi - self.pi)**2) < self.tol:
                self.pi = next_pi
                break
            self.pi = next_pi

        # Compute the scores
        self.scores = self.pi / self.ds_array
        # Normalize
        self.scores = self.scores / self.scores.sum()