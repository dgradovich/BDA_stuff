"""Call the GIbbs Samples from here"""
from Gibbs import Gibbs
from Stan import Modeller
import numpy as np
import pystan
import matplotlib.pyplot as plt
from Freq import Lin_reg, Probit_reg

class Inferencer:
    def __init__(self):
        # instantiate
        pass

    def gibbs_blr(self):
        x = np.arange(1,11);
        y = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0]);
        assert len(x) == len(y)

        tao_alpha = tao_beta = 1 / 100
        mu_alpha = mu_beta = 0
        tao = 1 / .25

        slope, intercept, r_value, p_value, std_err = Lin_reg.lin_reg(x, y)
        print(slope)
        print(intercept)

        # plot results
        # plt.scatter(x,y)
        # plt.plot(x, intercept + slope*x, c='r')
        # plt.show()

        # model = pystan.StanModel(model_code=Modeller.blr())
        # fit = model.sampling(data = {'N':10,'x':x,'y':y}, iter=1000, chains=2)
        # fit.plot()
        # print(fit)

        gibbs = Gibbs()
        intercept_g, slope_g = gibbs.gibbs_blr(tao_alpha, tao_beta, mu_alpha, mu_beta, x, y, tao)

        print(np.percentile(intercept_g, [10, 25, 50, 75, 90]))
        print(np.percentile(slope_g, [10, 25, 50, 75, 90]))

        plt.plot(intercept_g[100:]);
        plt.show()
        plt.plot(slope_g[100:]);
        plt.show()


if __name__ == '__main__':
    x = np.arange(1, 11)
    y = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    # prob = Probit_reg.probit_reg(x,y)
    # print(prob)
    model = pystan.StanModel(model_code=Modeller.probit_reg())
    fit = model.sampling(data = {'N':10,'x':x,'y':y}, iter=1000, chains=2)
    print(fit)