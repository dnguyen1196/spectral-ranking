from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.sparse.linalg import eigs
import numpy as np
import cvxpy as cp
import collections
from .pwlistorder import kwiksort

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
            p_hat = m * (-1 + k + np.exp(eps))/(np.exp(eps) - 1)\
                    - 1./(np.exp(eps)-1)
            return p_hat

        def recover_p_rappor(m, eps):
            p_hat = m * (1. + np.exp(eps/2.))/(np.exp(eps/2.)-1.)\
                    - 1./(np.exp(eps/2.)-1.)
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

        for (group, choices) in data_by_choice_group.items():
            t_group = tuple(group)
            all_items.update(t_group)
            L_Sa[t_group] = len(choices)
            k_min = min(k_min, len(group))

            group_choice[t_group] = {}
            for item in group:
                group_choice[t_group][item] = 0

            if self.mechanism == "rr":
                for y in choices:
                    # In rr, the randomized output is a single choice
                    group_choice[t_group][y] += 1
            else:
                # choice is shape (L, k)
                y_sum = np.sum(choices, axis=0)
                for i, item in enumerate(group):
                    group_choice[t_group][item] += y_sum[i]

        n = len(all_items)

        p_hat_arr = []

        for group, count in group_choice.items():
            group_items = list(group)
            m = np.array([count[item] for item in group_items])

            # Attempt to recover true winning probabilities
            if self.epsilon == np.inf:
                m = m/ m.sum()
                p_hat = m
            else:
                if self.mechanism == "rr":
                    m = m / m.sum()
                    p_hat = self.recover_p_rr(m, self.epsilon)
                else:
                    m /= L_Sa[group]
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
                    # group_choice[group][item] = p_hat_arr[a][i]
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
        constraints = [cp.sum(x) == 1, x >= delta] #, x <= 1- delta] <---------- Do we only need lower bound
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

            res = minimize(l1,x_0,args=y,constraints=linearconstraint)
            D_proj.append(res.x.tolist())
        return D_proj

class KwikSort():
    def __init__(self, epsilon, mechanism="rr", reg_l=0):
        self.epsilon = epsilon
        self.mechanism = mechanism
        self.reg_l = reg_l

    def fit(self, data_by_choice_group):
        # Aggregate the statistics
        self.aggregator = Aggregator(np.inf, reg_l=self.reg_l)

        self.group_choice, self.n, self.ds_array,\
            self.L_Sa\
                = self.aggregator.aggregate_raw_statistics(data_by_choice_group)

        list_els = [i for i in range(self.n)]
        pref_dict = self.get_preference_dict_generalized(self.group_choice, self.L_Sa)

        self.ranks = kwiksort(pref_dict, list_els, runs=1000, random_seed=None)
        # Run KwikSort on cmp data

    def fit_and_rank(self, data_by_choice_group):
        self.fit(data_by_choice_group)
        return self.ranks

    def get_ranks(self):
        return self.ranks

    def get_scores(self):
        return None

    def get_preference_dict(self, group_choice, L_Sa):
        pref_dict = {}
        for group, p in group_choice.items():
            sorted_group = sorted(list(group))
            item1 = sorted_group[0]
            item2 = sorted_group[1]
            sorted_group = tuple(sorted_group)
            assert(sorted_group not in pref_dict)
            L = L_Sa[group]

            pref_dict[sorted_group] = int(L * p[item1] - L*p[item2])

        return pref_dict

    def get_preference_dict_generalized(self, group_choice, L_Sa):
        pref_dict = collections.defaultdict(int)

        for group, p in group_choice.items():
            L = L_Sa[group]
            sorted_group = sorted(list(group))
            # Break down into pairwise comparisons here
            all_pairs = self.get_all_pairs(sorted_group)

            # How to deal with the case where the same pair appears
            # in different groups, just add?
            for pair in all_pairs:
                e1 = pair[0]
                e2 = pair[1]
                pref_dict[tuple(pair)] += int(L*p[e1] - L*p[e2])

        return pref_dict

    def get_all_pairs(self, ls):
        all_pairs = []
        for i in range(len(ls)-1):
            for j in range(i+1, len(ls)):
                all_pairs.append((ls[i], ls[j]))
        return all_pairs

    


