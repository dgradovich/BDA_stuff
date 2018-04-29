"""Metropolis Sampler (not MH)"""
import numpy as np
from scipy.stats import norm
from operator import mul
from functools import reduce
from tqdm import trange


class Metropolis:
    def __init__(self, c = 2.4):
        """Assume normal priors"""
        self.fixed_beta = 10
        self.prior_mean = 0
        self.prior_std = 100
        self.n = 5
        self.n_iter = 1000
        self.n_params = 2
        self.c = c
        self.sigma = np.diag([1,3])
        self.prior_mean = 0


    def metropolis_one_step(self,x,y):
        """Metropolis algorithm for Binomial model with fixed beta"""

        alpha = np.random.randn(1)

        for i in trange(self.n_iter):

            candidate = np.random.randn(1) + alpha[-1]
            candidate_prob = self._target_post(x,y,candidate, self.fixed_beta)
            alpha_candidate = alpha[-1]
            alpha_prob = self._target_post(x,y,alpha_candidate, self.fixed_beta)

            density_ratio = np.exp(candidate_prob - alpha_prob)
            density_ratio_prob = np.min([1, density_ratio])
            uni_n = np.random.uniform(0,1,1)

            if uni_n < density_ratio_prob:
                alpha = np.append(alpha, candidate)
            else:
                alpha = np.append(alpha, alpha_candidate)
        return alpha

    def _target_post(self,x,y, alpha, beta):
        """Evaluate target posterior - alpha is a parameter - binomial logit"""

        prior = norm(self.prior_mean,self.prior_std).pdf(beta)
        likelihood = (self._inv_logit(alpha + beta*x)**y)*((1-self._inv_logit(alpha + beta*x))**(self.n-y))
        likelihood = reduce(mul, likelihood)
        log_target = [np.log(each) for each in [prior, likelihood]]

        return np.sum(log_target)

    def _target_post_2d(self,x,y, alpha, beta):
        """Evaluate target posterior - alpha is a parameter - binomial logit"""
        prior_alpha = norm(self.prior_mean,self.prior_std).pdf(alpha)
        prior_beta = norm(self.prior_mean,self.prior_std).pdf(beta)
        likelihood = (self._inv_logit(alpha + beta*x)**y)*((1-self._inv_logit(alpha + beta*x))**(self.n-y))
        likelihood = reduce(mul, likelihood)

        log_target = [np.log(each) for each in [prior_alpha, prior_beta, likelihood]]
        # return prior_alpha * prior_beta * likelihood
        return np.sum(log_target)

    def _inv_logit(self, val):
        """Evaluate inverse logit"""
        return np.exp(val)/(1 + np.exp(val))

    def metropolis_two_step(self,x,y):
        """Metropolis algorithm for Binomial model - alpha and beta floating"""
        alpha = np.random.randn(1)
        beta = np.random.randn(1)

        for i in trange(self.n_iter):

            candidate = np.random.randn(1) + alpha[-1] # new candidate for alpha
            candidate_prob = self._target_post(x, y, candidate, beta[-1])
            old_alpha = alpha[-1]
            alpha_prob = self._target_post(x, y, old_alpha, beta[-1])
            density_ratio = np.exp(candidate_prob - alpha_prob)
            density_ratio_prob = np.min([1, density_ratio])
            uni_n = np.random.uniform(0, 1, 1)

            if uni_n < density_ratio_prob:
                alpha = np.append(alpha, candidate)
            else:
                alpha = np.append(alpha, old_alpha)

            candidate = np.random.randn(1) + beta[-1] # new candidate for beta
            candidate_prob = self._target_post(x, y, alpha[-1], candidate)
            old_beta = beta[-1]
            beta_prob = self._target_post(x, y, alpha[-1], old_beta)
            density_ratio = np.exp(candidate_prob - beta_prob)
            density_ratio_prob = np.min([1, density_ratio])
            uni_n = np.random.uniform(0, 1, 1)

            if uni_n < density_ratio_prob:
                beta = np.append(beta, candidate)
            else:
                beta = np.append(beta, old_beta)

        return alpha, beta

    def metropolis_2d(self,x,y):
        """Metropolis algorithm for Binomial model - 2d updates"""
        #init values for parameters
        params = [np.random.multivariate_normal(np.array([0,0]), np.diag([1,1]))]

        for i in trange(self.n_iter):

            candidate = np.random.multivariate_normal(np.array([0,0]), np.diag([1,1])) + params[-1] # new candidate for alpha
            candidate_prob = self._target_post_2d(x, y, candidate[0], candidate[1])
            old_params = params[-1]
            alpha_prob = self._target_post_2d(x, y, old_params[0], old_params[1])
            density_ratio = np.exp(candidate_prob - alpha_prob)
            density_ratio_prob = np.min([1, density_ratio])
            uni_n = np.random.uniform(0, 1, 1)

            if uni_n < density_ratio_prob:
                params.append(candidate)
            else:
                params.append(old_params)

        return [x[0] for x in params], [x[1] for x in params]

    def metropolis_2d_opt(self,x,y):
        """
        Metropolis algorithm for Binomial model - 2d updates - opt
        """
        #init values for parameters
        c = self.c / np.sqrt(self.n_params) # scaling param
        params = [np.random.multivariate_normal(np.array([0,0]), c**2*self.sigma)]

        for i in trange(self.n_iter):

            candidate = np.random.multivariate_normal(np.array([0,0]), c**2*self.sigma) + params[-1] # new candidate for alpha
            candidate_prob = self._target_post_2d(x, y, candidate[0], candidate[1])
            old_params = params[-1]
            alpha_prob = self._target_post_2d(x, y, old_params[0], old_params[1])
            density_ratio = candidate_prob / alpha_prob
            density_ratio_prob = np.min([1, density_ratio])
            uni_n = np.random.uniform(0, 1, 1)

            if uni_n < density_ratio_prob:
                params.append(candidate)
            else:
                params.append(old_params)

        return [x[0] for x in params], [x[1] for x in params]

