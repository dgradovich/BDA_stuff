"""Slice sampling updates for alpha and beta in a binom model"""

import numpy as np
from Metropolis import Metropolis

class Slicer(Metropolis):
    def __init__(self):
        Metropolis.__init__(self)

    def _update_alpha(self):
        """For a fixed beta update alpha"""
        x = np.random.uniform(0,1,1)

        pass

    def _update_beta(self):
        """For a fixed alpha update beta"""
        pass


    def slice_sampling(self):
        """Slice sampling"""
        params = [np.random.multivariate_normal(np.array([0,0]), self.sigma)]


        pass