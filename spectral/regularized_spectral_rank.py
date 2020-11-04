import numpy as np
import cvxpy as cp
import collections
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

class Aggregator():
    """ Aggregate the choice data to produce 
    (choice_set, statistics) tuples

    """
    def __init__(self, epsilon, mechanism="rr", reg_l=0.):
        self.epsilon = epsilon
        self.mechanism = mechanism
        self.reg_l = reg_l
        
        def recover_p_krr(m, eps):
            k = len(m)
            p_hat = m * (1 + k + np.exp(eps))/(np.exp(eps) - 1)\
                    - 1./(np.exp(eps)-1)
            return p_hat

        def recover_p_rappor(m, eps):
            p_hat = m * (1 + np.exp(eps/2))/(np.exp(eps/2)-1)\
                    - 1./(np.exp(eps/2)-1)
            return p_hat

        self.recover_p_rr = recover_p_krr
        self.recover_p_rappor = recover_p_rappor

    def aggregate_raw_statistics(self, data_by_choice_group):
        L_Sa = collections.defaultdict(int) # {group_choice_a -> La}
        group_choice = {} # {group -> p_Sa}
        N = len(data_by_choice_group.keys())
        L = max([len(choice) for (group, choice) in data_by_choice_group.items()])

        # Keep all the items in the universe
        all_items = set()
        k_min = np.inf

        # This loop is expensive
        # for user_data in data_by_choice_group:
        #     for (group, y) in user_data:
        #         t_group = tuple(group)
        #         k_min = min(k_min, len(group))

        #         L_Sa[t_group] += 1
        #         all_items.update(t_group)

        #         if t_group not in group_choice:
        #             group_choice[t_group] = {}
        #             for item in group:
        #                 group_choice[t_group][item] = 0

        #         # In rr, the randomized output is a single choice
        #         if self.mechanism == "rr":
        #             group_choice[t_group][y] += 1
        #         else:
        #         # In rappor, the randomized output is a vector
        #             for i, item in enumerate(group):
        #                 group_choice[t_group][item] += y[i]

        for (group, choices) in data_by_choice_group.items():
            t_group = tuple(group)
            all_items.update(t_group)
            L_Sa[t_group] = len(choices)
            k_min = min(k_min, len(group))

            group_choice[t_group] = {}
            for item in group:
                group_choice[t_group][item] = 0

            # This for loop could be fastened / removed
            for y in choices:
                # In rr, the randomized output is a single choice
                if self.mechanism == "rr":
                    group_choice[t_group][y] += 1
                else:
                # In rappor, the randomized output is a vector
                    for i, item in enumerate(group):
                        group_choice[t_group][item] += y[i]

        n = len(all_items)

        p_hat_arr = []

        for group, count in group_choice.items():
            group_items = list(group)
            m = np.array([count[item] for item in group_items])
            m = m/ L_Sa[group]

            # Attempt to recover true winning probabilities
            if self.epsilon == np.inf:
                p_hat = m
            else:
                if self.mechanism == "rr":
                    p_hat = self.recover_p_rr(m, self.epsilon)
                else:
                    p_hat = self.recover_p_rappor(m, self.epsilon)

            p_hat_arr.append(p_hat)

        if self.epsilon == np.inf:
            for a, group in enumerate(group_choice.keys()):
                for i, item in enumerate(list(group)):
                    group_choice[group][item] = p_hat_arr[a][i]
        else:
            D_proj = self.project(p_hat_arr, np.sqrt(np.log2(n))/(L * k_min**2))
            for a, group in enumerate(group_choice.keys()):
                for i, item in enumerate(list(group)):
                    group_choice[group][item] = D_proj[a][i]
        
        ds_array = np.zeros((n,))
        for group in L_Sa.keys():
            for i in group:
                ds_array[i] += L_Sa[group]/len(group) + self.reg_l

        return group_choice, n, ds_array, L_Sa
       
    def project_to_probability_simplex(self, v, delta=1e-4):
        """ Project a vector onto the probability simplex by solving the
        quadratic program:

        minimize || x - v ||_1
        
        such that: 
                x >= 0
                x @ 1 = 1
        """
        k = len(v)
        x = cp.Variable(k)

        objective = cp.Minimize(0.5 * cp.norm1(x-v))
        constraints = [cp.sum(x) == 1, x >= delta, x <= 1- delta]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Remove negative numbers and normalize
        x_hat = np.maximum(x.value, 0)
        x_hat = x_hat / x_hat.sum() 
        return x_hat

    def project(self, D_u, delta):
        def l1(x,y):
            return np.linalg.norm(x-y,1)

        D_proj = []
        for d_u in D_u:
            k = len(d_u)
            A = []
            A.append([1]*k)
            l = [1]
            u = [1]
            for i in range(k):
                a = [0]*k
                a[i] = 1
                A.append(a)
                l.append(delta)
                u.append(1-delta)
            linearconstraint = LinearConstraint(A,l,u)
            x_0 = np.array(d_u)
            y = np.array(d_u)
            for i in range(k):
                if d_u[i] > 1: 
                    x_0[i] = 1
                elif d_u[i] < 0: 
                    x_0[i] = 0
            res = minimize(l1,x_0,args=y,constraints=linearconstraint)
            D_proj.append(res.x.tolist())
        return D_proj

class RegularizedSpectralRank():
    def __init__(self, epsilon, mechanism="rr", reg_l=0, max_iters=10000, tol=1e-12):
        """

        Attributes:

        """
        self.epsilon = epsilon
        self.mechanism = mechanism
        self.reg_l = reg_l
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, data_by_choice_group):
        """ Learn the BTL/MNL model
        """
        # Aggregate the statistics
        self.aggregator = Aggregator(self.epsilon, self.mechanism, self.reg_l)

        self.group_choice, self.n, self.ds_array,\
            self.L_Sa\
                = self.aggregator.aggregate_raw_statistics(data_by_choice_group)

        # Construct matrix P
        self.construct_P()
        # Do power iteration
        self.find_stationary_distribution()

    def fit_and_rank(self, data_by_choice_group):
        """ 
        Return the list of items, from highest score to lowest score
        """
        self.fit(data_by_choice_group)
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
        self.P = np.zeros((self.n, self.n))
        d = self.ds_array

        for (Sa, p_Sa) in self.group_choice.items():
            for i in Sa:
                for j in Sa:
                    # P'ij = 1/d'[i] sum (n_j|Sa + lambda)/|Sa|
                    self.P[i, j] += 1./d[i] *\
                        (p_Sa[j] * self.L_Sa[Sa] + self.reg_l)/len(Sa)
            
    def find_stationary_distribution(self):
        """ Compute the stationary distribution and the scores
        """
        # Start with a uniform distribution
        self.pi = np.ones((self.n,)) / self.n

        for i in range(self.max_iters):
            next_pi = np.matmul(self.P.T, self.pi)
            next_pi /= next_pi.sum()
            if np.sum((next_pi - self.pi)**2) < self.tol:
                self.pi = next_pi
                break
            self.pi = next_pi
        
        # Compute the scores
        self.scores = self.pi / self.ds_array
        # Normalize
        self.scores = self.scores / self.scores.sum()