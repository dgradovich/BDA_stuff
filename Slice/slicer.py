"""Slice sampling updates for alpha and beta in a binom model"""

import numpy as np
from Metropolis import Metropolis
from tqdm import trange
import logging


class Slicer(Metropolis):
    def __init__(self):
        Metropolis.__init__(self)
        self.w = 1
        self.n_iter = 500

    def _update_param(self,x,y, parameters, position):
        """Update parameter for one sampling
        :param position {0,1} - position of a parameter being updated
        :param parameters most recent update for alpha and beta
        """
        param_prop = np.random.uniform(0,1)
        parameters[position] = param_prop
        y_sample = np.random.uniform(0, np.exp(self._target_post_2d(x, y, parameters[0], parameters[1])))
        U = np.random.uniform(0, 1)
        L = param_prop - U * self.w
        R = param_prop + self.w * (1 - U)

        """Increase the interval length"""

        parameters[position] = L
        while self._target_post_2d(x, y, parameters[0], parameters[1]) > y_sample:
            L = L - self.w
            parameters[position] = L

        parameters[position] = R
        while self._target_post_2d(x, y, parameters[0], parameters[1]) > y_sample:
            R = R + self.w
            parameters[position] = R

        param_prop = np.random.uniform(L, R)
        parameters[position] = param_prop

        """Acceptance from the interval"""

        accept = False
        while accept == False:
            if self._target_post_2d(x, y, parameters[0], parameters[1]) > y_sample:
                accept = True
                return parameters
            else:
                if np.abs(parameters[position] - L) < np.abs(parameters[position] - R):
                    L = parameters[position]
                else:
                    L = np.exp(y_sample)

                param_prop = np.random.uniform(low=L, high=R)
                parameters[position] = param_prop

                logging.info('Not accepted')

    def slice_sampling(self,x,y):
        """Slice sampling"""
        # params = [np.random.multivariate_normal(np.array([0,0]), self.sigma)] #alpha and beta
        params = [np.random.uniform(0,1,2)]
        for i in trange(1,self.n_iter):
            params[i] = self._update_param(x,y, params[-1],0) # update alpha
            params[i] = append(self._update_param(x,y, params[-1],1)) # update beta
        return [x[0] for x in params], [x[1] for x in params]

    def alpha_ss(self,x,y):
        w = 1
        # x_prev = np.random.uniform(0,1,1)
        params = [np.random.uniform(0,1,2)]

        trace = []
        for k in trange(self.n_iter):
            x_prev = params[-1][0]
            slope_prev = params[-1][1]

            y_samp = np.random.uniform(low=0, high=np.exp(self._target_post_2d(x,y,x_prev, slope_prev)))
            U = np.random.rand()
            L = x_prev - U * w
            R = x_prev + w * (1.0 - U)
            while np.exp(self._target_post_2d(x,y,L, slope_prev)) > y_samp:
                L = L - w
            while np.exp(self._target_post_2d(x,y,R, slope_prev)) > y_samp:
                R = R + w

            x_prop = np.random.uniform(low=L, high=R)

            accept = False
            while accept == False:
                if y_samp < np.exp(self._target_post_2d(x,y,x_prop, slope_prev)):
                    x_prev = x_prop
                    # trace.append(x_prop)
                    accept = True
                else:  # propose again: in real slice we would shrink
                    x_prop = np.random.uniform(low=L, high=R)

            y_samp = np.random.uniform(low=0, high=np.exp(self._target_post_2d(x,y,x_prev, slope_prev)))
            U = np.random.rand()
            L = slope_prev - U * w
            R = slope_prev + w * (1.0 - U)
            while np.exp(self._target_post_2d(x,y,x_prev, L)) > y_samp:
                L = L - w
            while np.exp(self._target_post_2d(x,y,x_prev, R)) > y_samp:
                R = R + w

            slope_prop = np.random.uniform(low=L, high=R)

            accept = False
            while accept == False:
                if y_samp < np.exp(self._target_post_2d(x,y,x_prev, slope_prop)):
                    slope_prev = slope_prop
                    # trace.append(x_prop)
                    accept = True
                else:  # propose again: in real slice we would shrink
                    slope_prop = np.random.uniform(low=L, high=R)

            trace.append(np.array([x_prev, slope_prev]))
        return [x[0] for x in trace], [x[1] for x in trace]