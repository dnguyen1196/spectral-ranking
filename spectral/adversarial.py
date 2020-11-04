# An adversarial example that can recover (albeit in exponential time) the preference data of 
# L-th user given preference data of L-1 users
import numpy as np
import copy
from copy import deepcopy
import queue
import collections
import time

class Adversary():
    def __init__(self, w, other_data, 
                    choice_groups, reg_l=0.):
        """
        Parameters:

        other_data = array of choice data of length L-1
            other_data[i] is an array of (group, choice) data for user i
            Assume that the group is the same for all the users

        Since we are enumerating all the possible choices made by the Lth user
        What is a smart way of testing each configuration?

        """
        self.w = w # Final item scores
        self.other_data = other_data
        self.choice_groups = choice_groups
        self.reg_l = 0.
        self.n = len(w)

        self.L_Sa = collections.defaultdict(int) # choice_group -> number of users given
        for user_data in other_data:
            for choice_group, winner in user_data:
                self.L_Sa[tuple(choice_group)] += 1

        self.d = np.zeros((self.n,))
        for group in self.choice_groups:
            for i in group:
                self.d[i] += (self.L_Sa[tuple(group)]+1)/len(group)
                # self.d[i] += 1
        
        self.pi = self.d * self.w
        self.pi /= self.pi.sum() # Normalize
        self.N = len(self.choice_groups) # Number of choice groups
        self.L = len(self.other_data) + 1 # Total number of users

    def recover_user_data(self, all_choices=None, choice_groups=None):
        # Given the all the possible choices the user can make
        # Check each one
        minerr = np.inf
        bestguess = None

        if all_choices is None:
            all_choices = self.enumerate_all_choices_dfs()


        if not hasattr(self, "P_star_mat"):
            self.P_star, self.L_star = self.compute_winning_probabilities()
            self.P_star_mat = self.compute_P_star_mat(self.P_star, self.L_star)

        if choice_groups is not None:
            self.choice_groups = choice_groups

        matched = []

        # Now check each choice
        start = time.time()
        for i in range(len(all_choices)):
            guess = all_choices[i]

            # P = self.compute_P_matrix(guess, P_star, L_star)
            P = self.compute_P_matrix_from_P_star_matrix(
                guess, deepcopy(self.P_star_mat), self.L_star)

            err = np.linalg.norm(self.pi - P.T @ self.pi)

            if err < minerr:
                minerr = err
                bestguess  = guess

            if err < 1e-5:
                matched.append(guess)

        return matched, minerr, bestguess

    def compute_P_star_mat(self, P_star, L_star):
        P_star_mat = np.zeros((self.n, self.n))
        d = self.d
        for a, group in enumerate(self.choice_groups):
            k = tuple(group)
            for i in group:
                for j in group:
                    try:
                        # P_star_mat[i, j] += 1./d[i] * P_star[k][j]\
                        #     * (L_star[k])/(L_star[k]+1)
                        P_star_mat[i, j] += 1./d[i] *\
                            (P_star[k][j] * L_star[k] + self.reg_l)/len(group)

                    except Exception as e:
                        # print(i)
                        # print(j)
                        # print(k)
                        # print(P_star)
                        # print(L_star)
                        # raise Exception
                        continue
        return P_star_mat

    def enumerate_all_choices_dfs(self, choice_groups=None):
        # Enumerate all possible choices via DFS instead of BFS
        # all_choices = list[user_choice]
        # user_choice = list[winner]
        # Note that user_choice order follows the same as group_choices
        if choice_groups is None:
            choice_groups = self.choice_groups
        all_choices = []

        def recurse(all_choices, current_choice):
            # Base case
            if len(current_choice) == len(choice_groups):
                all_choices.append(current_choice)
                return

            a = len(current_choice)
            for winner in choice_groups[a]:
                recurse(all_choices, current_choice + [winner])

        recurse(all_choices, [])
        return all_choices

    def compute_winning_probabilities(self):
        """

        We want to compute 
            P_{i|Sa} -> Empirical winning probabilities
            L_a -> Number of users contributing to choices related to group Sa

        """
        # L: choice_set -> number of votes
        L = collections.defaultdict(int)

        # P_S: choice_set -> {item -> winning counts}
        P_S = collections.defaultdict(dict)

        for i, user_data in enumerate(self.other_data):
            for choice_group, winner in user_data:
                L[tuple(choice_group)] += 1
                if winner not in P_S[tuple(choice_group)]:
                    P_S[tuple(choice_group)][winner] = 0
                P_S[tuple(choice_group)][winner] += 1

        for (choice_set, winning_counts) in P_S.items():
            total_counts = sum([count for winner, count in winning_counts.items()])
            for winner in winning_counts.keys():
                winning_counts[winner] *= 1./total_counts

            for item in choice_set:
                if item not in winning_counts:
                    winning_counts[item] = 0.

        return P_S, L

    def compute_P_matrix_from_P_star_matrix(self, user_choice, P_star_mat, L_star):
        # Recompute the P matrix when given the L-th user's private data
        # NOTE: assume that user_choice list in the same order as choice_groups
        choice_groups = self.choice_groups
        for a, winner in enumerate(user_choice):
            group = self.choice_groups[a]
            k = tuple(group)
            if winner != -1:
                for i in group:
                    # P_star_mat[i, winner] += 1./self.d[i] * 1./(L_star[k]+1)
                    P_star_mat[i, winner] += 1./self.d[i] * 1./len(group)

        return P_star_mat

    def update_scores(self, new_score):
        self.w = new_score/np.sum(new_score)
        self.pi = self.d * self.w
        self.pi /= self.pi.sum() # Normalize
