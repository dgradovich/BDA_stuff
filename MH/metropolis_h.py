"""Metropolis Hastings"""

import numpy as np
from Metropolis import Metropolis
from PE import PointEstimator
from tqdm import trange
from scipy.stats import multivariate_normal

class MH(Metropolis):

    def __init__(self,x,y):
        Metropolis.__init__(self)
        self.x = x
        self.y = y
        self.MAP = self._get_map()
        self.delta = 2
        self.sigma = self.c ** 2 * np.diag([np.sqrt(self.MAP[0]),np.sqrt(self.MAP[1])])
        # self.n_iter = 2000

    def _candidate_direction(self, param_val):
        """Compute the direction of the jump"""
        return param_val + self.delta * (self.MAP - param_val)/np.linalg.norm(self.MAP - param_val)

    def _get_map(self):
        pe = PointEstimator(self.x, self.y)
        return pe.MAP().x

    def perform_mh(self):
        """Metropolis-Hastings"""
        # c = self.c / np.sqrt(self.n_params) # scaling param
        params = [np.random.multivariate_normal(np.array([0,0]), np.diag([1,1]))]



        for i in trange(self.n_iter):

            candidate = np.random.multivariate_normal(np.array([0,0]), self.sigma) + self._candidate_direction(params[-1]) # new candidate for alpha
            candidate_prob = self._target_post_2d(self.x, self.y, candidate[0], candidate[1])
            j_candidate = np.log(multivariate_normal(self._candidate_direction(params[-1]), self.sigma).pdf(candidate))

            old_params = params[-1]
            alpha_prob = self._target_post_2d(self.x, self.y, old_params[0], old_params[1])
            j_alpha = np.log(multivariate_normal(self._candidate_direction(candidate), self.sigma).pdf(params[-1]))
            # print(j_candidate < j_alpha)

            # density_ratio = np.exp(candidate_prob - j_candidate - alpha_prob + j_alpha)
            density_ratio = np.exp(candidate_prob-alpha_prob)


            density_ratio_prob = np.min([1, density_ratio])
            uni_n = np.random.uniform(0, 1, 1)

            if uni_n < density_ratio_prob:
                params.append(candidate)
            else:
                params.append(old_params)

        return [x[0] for x in params], [x[1] for x in params]