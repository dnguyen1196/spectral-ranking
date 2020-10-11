# An adversarial example that can recover (albeit in exponential time) the preference data of 
# L-th user given preference data of L-1 users
import numpy as np
import copy
from copy import deepcopy
import queue

class Adversary():
    def __init__(self, w, other_data, N=None, k=None, L=None, n=None):
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
        self.N = len(other_data[0]) if not N else N # Number of choice groups
        self.k = len(other_data[0][0][0]) if not k else k # Size of choice groups
        self.L = len(other_data) + 1 if not L else L # Number of users

        self.choice_groups = [ # Get all the choice groups
            # Each choice_data is a (group, choice) tuple
            choice_data[0] for choice_data in other_data[0]
        ]
        if n is not None:
            self.n = n
        else:
            universe = set()
            for g in self.choice_groups:
                for item in g:
                    universe.add(item)
            self.n = len(universe)
        
        self.d = np.zeros((self.n,))
        for group in self.choice_groups:
            for i in group:
                self.d[i] += 1
        
        self.pi = self.d * self.w
        self.pi /= self.pi.sum() # Normalize

    def recover_user_data(self, all_choices=None):
        # Given the
        minerr = np.inf
        bestguess = None

        if not all_choices:
            all_choices = self.enumerate_all_choice()
        matched = []
        self.compute_P_star_matrix()

        for guess in all_choices:
            P = self.compute_P_matrix(guess)
            err = np.linalg.norm(self.pi - P.T @ self.pi)

            if err < minerr:
                minerr = err
                bestguess  = guess

            if err < 1e-4:
                matched.append(guess)

        return matched, minerr, bestguess

    def enumerate_all_choice(self):
        # Enumerate all the possible choice that the Lth user could make
        # We will have N^k possible choices the L-th user can ever make
        choice_groups = self.choice_groups
        
        all_choices = []
        q = queue.Queue()

        for item in choice_groups[0]:
            q.put([item])
        # Do a BFS-like sweep or a recursive DFS

        while q.qsize() > 0:
            head = q.get()
            if len(head) == self.N:
                all_choices.append(head)
            else:
                a = len(head) # Pick a winning item for the next choice group
                # Note that it's 0-indexing so a -> index of the next choice group
                for item in choice_groups[a]:
                    q.put(head + [item])

        return all_choices

    def compute_P_star_matrix(self):
        # Compute the transition matrix from the data of the first L-1 users
        self.P_star = [] 
        # This should be an array of (group, p_Sa) where p_Sa is a dictionary

        for a in range(self.N):
            choice_group = self.choice_groups[a]
            p_Sa = dict([(item, 0) for item in choice_group])
            for l in range(self.L-1):
                winning_item = self.other_data[l][a][1]
                p_Sa[winning_item] += 1./(self.L-1)
            self.P_star.append((choice_group, p_Sa))

    def compute_P_matrix(self, user_choice):
        # Recompute the P matrix when given the L-th user's private data
        P = np.zeros((self.n, self.n))
        d = self.d

        for a, (group, p_Sa) in enumerate(self.P_star):
            winning_item = user_choice[a]
            for i in group:
                for j in group:
                    P[i, j] += 1./d[i] * p_Sa[j] * (self.L-1)/self.L
                P[i, winning_item] += 1./d[i] * 1./self.L

        return P