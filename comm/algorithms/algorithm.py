__doc__ = """
Base algorithm class
"""

import numpy as np
from comm._rod_tool import StaticRod

class Algorithm:  # TODO: maybe rename the class??
    """Algorithm.
    """

    def __init__(self, rod, algo_config):
        """__init__.

        Parameters
        ----------
        rod :
        algo_config :
        """

        self.static_rod = StaticRod.get_rod(rod)
        self.config = algo_config

        self.ds = self.static_rod.rest_lengths / np.sum(self.static_rod.rest_lengths)
        self.s = np.insert(np.cumsum(self.ds), 0, 0)
        self.s_position = self.s.copy()
        self.s_director = (self.s[:-1] + self.s[1:])/2
        self.s_sigma = (self.s[:-1] + self.s[1:])/2
        self.s_kappa = self.s[1:-1]

    def run(self, **kwargs):
        """run.

        Parameters
        ----------
        kwargs :
        """
        raise NotImplementedError
