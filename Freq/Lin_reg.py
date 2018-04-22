"""Frequentist methods e.g. simple linear reg for comparison"""
import numpy as np
from scipy.stats import linregress


class Lin_reg:
    def __init__(self):
        pass

    @staticmethod
    def lin_reg(x,y):
        return linregress(x,y)