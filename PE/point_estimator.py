"""Point estimates with optimisation"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from functools import reduce
from operator import mul
from Metropolis import Metropolis

class PointEstimator:
    def __init__(self,x,y):
        self.prior_mean = 0
        self.prior_var = 100
        self.x = x
        self.y = y
        self.n = 5


    def MAP(self):
        """Find a posterior mode"""
        initial_guess = np.random.multivariate_normal(np.array([0,0]), np.diag([1,1]))
        mode_prediction = minimize(self._lkhood, initial_guess)
        return mode_prediction

    def _lkhood(self, params):
        """Logistic model posterior"""
        m = Metropolis()
        alpha, beta = params
        alpha_component = norm(self.prior_mean,self.prior_var).pdf(alpha)
        beta_component = norm(self.prior_mean,self.prior_var).pdf(beta)
        likelihood = (m._inv_logit(alpha + beta*self.x)**self.y)*((1-m._inv_logit(alpha + beta*self.x))**(self.n-self.y))
        likelihood = reduce(mul, likelihood)
        return -1*np.log(alpha_component * beta_component * likelihood) # - log probability for stability


