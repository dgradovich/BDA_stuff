"""Gibbs sampler"""
import numpy as np
from tqdm import trange

class Gibbs:
    def __init__(self):
        self.gibs_iter = 500

    def gibbs_blr(self,tao_alpha, tao_beta, mu_alpha, mu_beta, x, y, tao):
        """Gibbs Sampler for Bayesian Linear Regression"""
        N = len(x)
        alpha_var = 1/(tao_alpha+tao*N)
        beta_var = 1/(tao_beta+tao*np.sum(x**2))
        alpha = 1/tao_alpha * np.random.randn(1) + mu_alpha
        beta = 1/tao_beta * np.random.randn(1) + mu_beta

        for i in trange(self.gibs_iter):
            alpha_candidate = (tao_alpha*mu_alpha + tao*np.sum(y-beta[-1]*x))*alpha_var + np.random.randn(1)*alpha_var
            alpha = np.append(alpha, alpha_candidate)

            beta_candidate = (tao_beta*mu_beta + tao*np.sum((y-alpha[-1])*x))*beta_var + np.random.randn(1)*beta_var
            beta = np.append(beta, beta_candidate)

        return alpha, beta


    @staticmethod
    def gibbs_probit_reg():
        """Gibbs Sampler for Probit regression"""
        N = len(x)
