"""Call the GIbbs Samples from here"""
from Gibbs import Gibbs
from Stan import Modeller
import numpy as np
import pystan
import matplotlib.pyplot as plt
from Freq import Lin_reg, Probit_reg
from Metropolis import Metropolis
from PE import PointEstimator
from pymc3.stats import autocorr
from Slice import Slicer
from MH import MH

class Inferencer:
    def __init__(self):
        self.burn_in = 250

    def autocorr(self, posterior_chain):
        """Produce autocorrelation plots"""
        lags = np.arange(1, 250)
        plt.plot(lags, [autocorr(posterior_chain[self.burn_in:], l) for l in lags]);
        plt.show()

    @staticmethod
    def gibbs_blr():
        """Bayesian linear regression with Gibbs"""
        # Toy data
        x = np.arange(1,11);
        y = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0]);
        assert len(x) == len(y)
        # set parameters
        tao_alpha = tao_beta = 1 / 100
        mu_alpha = mu_beta = 0
        tao = 1 / .25

        # Non-bayesian implementation
        slope, intercept, r_value, p_value, std_err = Lin_reg.lin_reg(x, y)
        print(slope)
        print(intercept)

        plt.scatter(x,y)
        plt.plot(x, intercept + slope*x, c='r')
        plt.show()

        # Stan implementation
        model = pystan.StanModel(model_code=Modeller.blr())
        fit = model.sampling(data = {'N':10,'x':x,'y':y}, iter=1000, chains=2)
        fit.plot()
        print(fit)

        # Gibbs
        gibbs = Gibbs()
        intercept_g, slope_g = gibbs.gibbs_blr(tao_alpha, tao_beta, mu_alpha, mu_beta, x, y, tao)

        print(np.percentile(intercept_g, [10, 25, 50, 75, 90]))
        print(np.percentile(slope_g, [10, 25, 50, 75, 90]))

        plt.plot(intercept_g[100:]);
        plt.show()
        plt.plot(slope_g[100:]);
        plt.show()

    @staticmethod
    def gibbs_probit():
        """Probit regression with Gibbs"""

        # Toy data
        x = np.arange(1, 11)
        y = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])

        # Parameter values
        tao_alpha = tao_beta = 1 / 100
        mu_alpha = mu_beta = 0

        # Non-bayesian implementation
        prob = Probit_reg.probit_reg(x, y)
        print(prob)

        # Stan
        model = pystan.StanModel(model_code=Modeller.probit_reg())
        fit = model.sampling(data = {'N':10,'x':x,'y':y}, iter=1000, chains=2)
        print(fit)

        # Gibbs TODO
        gibbs = Gibbs()
        intercept_g, slope_g = gibbs.gibbs_probit_reg(tao_alpha, tao_beta, mu_alpha, mu_beta, x, y)

        print(np.percentile(intercept_g, [10, 25, 50, 75, 90]))
        print(np.percentile(slope_g, [10, 25, 50, 75, 90]))


    def binom(self):
        """Logistic with Metropolis"""
        # Toy data
        n = 5*np.ones(4)
        x = np.array([-0.86, -0.30, -0.05, 0.73])
        y = np.array([0,1,3,5])

        # Parameter values
        tao_alpha = tao_beta = 1/100**2

        # model = pystan.StanModel(model_code=Modeller.logistic_reg())
        # fit = model.sampling(data = {'N':len(x),'x':x,'y':y}, iter=1000, chains=2)
        # print(fit)
        #
        # model = pystan.StanModel(model_code=Modeller.logistic_reg_fixed_beta())
        # fit = model.sampling(data = {'N':len(x),'x':x,'y':y, 'beta':10}, iter=1000, chains=2)
        # print(fit)
        #
        # m = Metropolis()
        # alpha_est = m.metropolis_one_step(x,y)
        # print(np.percentile(alpha_est[500:], [10, 25, 50, 75, 90]))

        # m = Metropolis()
        # alpha_est, beta_est = m.metropolis_two_step(x,y)
        # print(np.percentile(alpha_est[self.burn_in:], [10, 25, 50, 75, 90]))
        # print(np.percentile(beta_est[self.burn_in:], [10, 25, 50, 75, 90]))
        #
        # plt.scatter(alpha_est[self.burn_in:], beta_est[self.burn_in:]);plt.show()
        # plt.hist(-1 * alpha_est[self.burn_in:] / beta_est[self.burn_in:]); plt.show() # LD50 values
        #
        # m = Metropolis()
        # alpha_est, beta_est = m.metropolis_2d(x,y)
        # print(np.percentile(alpha_est[self.burn_in:], [10, 25, 50, 75, 90]))
        # print(np.percentile(beta_est[self.burn_in:], [10, 25, 50, 75, 90]))
        #
        # plt.scatter(alpha_est[self.burn_in:], beta_est[self.burn_in:]);plt.show()
        # plt.hist(-1 * np.array(alpha_est)[self.burn_in:] / np.array(beta_est)[self.burn_in:]); plt.show() # LD50 values

        # m = Metropolis()
        # alpha_est, beta_est = m.metropolis_2d_opt(x,y)
        # print(np.percentile(alpha_est[self.burn_in:], [10, 25, 50, 75, 90]))
        # print(np.percentile(beta_est[self.burn_in:], [10, 25, 50, 75, 90]))

        # plt.scatter(alpha_est[self.burn_in:], beta_est[self.burn_in:]);plt.show()
        # plt.hist(-1 * np.array(alpha_est)[self.burn_in:] / np.array(beta_est)[self.burn_in:]); plt.show() # LD50 values
        #
        # mh = MH(x,y)
        # alpha_est, beta_est = mh.perform_mh()
        # print(np.percentile(alpha_est[self.burn_in:], [10, 25, 50, 75, 90]))
        # print(np.percentile(beta_est[self.burn_in:], [10, 25, 50, 75, 90]))
        #
        # self.autocorr(alpha_est[self.burn_in:])
        # self.autocorr(beta_est[self.burn_in:])
        # plt.scatter(alpha_est[self.burn_in:], beta_est[self.burn_in:]);plt.show()
        #
        # plt.show()

        s = Slicer()
        # alpha_est, beta_est = s.alpha_ss(x,y)
        # print(np.percentile(alpha_est[self.burn_in:], [10, 25, 50, 75, 90]))
        # print(np.percentile(beta_est[self.burn_in:], [10, 25, 50, 75, 90]))
        alpha_est, beta_est = s.slice_sampling(x,y)
        print(np.percentile(alpha_est[self.burn_in:], [10, 25, 50, 75, 90]))
        print(np.percentile(beta_est[self.burn_in:], [10, 25, 50, 75, 90]))

if __name__ == '__main__':
    inference = Inferencer()
    inference.binom()
