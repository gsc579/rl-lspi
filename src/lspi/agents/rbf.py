import itertools

import numpy as np
from lspi.agents.agent import Agent


class RadialAgent(Agent):
    def __init__(self, env, centers, sigma=1., preprocess_obs=None):
        self.centers = centers
        self.sigma2 = sigma**2
        super(RadialAgent, self).__init__(env, preprocess_obs)

    def _get_features(self, obs):
        dists = np.power(self.centers - obs, 2)#power(x, y) 函数，计算 x 的 y 次方
        rbfs = np.exp(-dists.sum(1) / (2 * self.sigma2))
        return np.append(rbfs, [1.])

    @staticmethod
    def get_centers_from_grids(grids):
        return np.array(list(itertools.product(*grids)))
