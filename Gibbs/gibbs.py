"""Gibbs sampler"""
import numpy as np
from tqdm import trange
from scipy.stats import truncnorm

class Gibbs:
    def __init__(self):
        self.gibs_iter = 1000

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

            beta_candidate = (tao_beta*mu_beta + tao*np.sum(x*((y-alpha[-1]))))*beta_var + np.random.randn(1)*beta_var
            beta = np.append(beta, beta_candidate)

        return alpha, beta

    def gibbs_probit_reg(self,tao_alpha, tao_beta, mu_alpha, mu_beta, x, y):
        """Gibbs Sampler for Probit regression"""
        # Set params
        # N = len(x)
        # alpha_var = 1 / (tao_alpha + N)
        # beta_var = 1 / (tao_beta + np.sum(x**2))
        #
        # #Init variables
        # alpha = 1 / tao_alpha * np.random.randn(1) + mu_alpha
        # # print(alpha)
        # beta = 1 / tao_beta * np.random.randn(1) + mu_beta
        # # print(beta)
        # z = [np.array(truncnorm.rvs(a = 0,b = np.inf, loc = alpha[-1] + beta[-1]*x[i], scale = 1)) if y[i] == 1 else \
        #          np.array(truncnorm.rvs(a = -np.inf,b = 0, loc = alpha[-1] + beta[-1]*x[i], scale = 1)) for i in range(N)]
        #
        # # print(z)
        #
        # for i in trange(self.gibs_iter):
        #     alpha_candidate = (tao_alpha * mu_alpha + np.sum(z[-1]-beta[-1]*x)) * alpha_var + np.random.randn(1) * alpha_var
        #     alpha = np.append(alpha, alpha_candidate)
        #
        #     beta_candidate = (tao_beta * mu_beta + np.sum(x*(z[-1]-alpha[-1]))) * beta_var + np.random.randn(1) * beta_var
        #     beta = np.append(beta, beta_candidate)
        #
        #     z_candidate = [np.array(truncnorm.rvs(a = 0,b = np.inf, loc = alpha[-1] + beta[-1]*x[i], scale = 1)) if y[i] == 1 else \
        #          np.array(truncnorm.rvs(a = -np.inf,b = 0, loc = alpha[-1] + beta[-1]*x[i], scale = 1)) for i in range(N)]
        #     z.append(z_candidate)

        N = len(x)
        x = np.append(np.ones(10).reshape(-1,1),x.reshape(-1,1), axis = 1).reshape(len(x),2)



        return params

