"""Prboit regression"""
from statsmodels.discrete.discrete_model import Probit
import numpy as np

class Probit_reg:
    def __init__(self):
        pass

    @staticmethod
    def probit_reg(x,y):
        """Univariate probit regression"""
        x = np.append(np.ones(10).reshape(-1,1),x.reshape(-1,1), axis = 1).reshape(len(x),2)
        pm = Probit(y,x)
        return pm.fit().params