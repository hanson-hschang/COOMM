"""
Created on Nov. 14, 2021
@author: Heng-Sheng (Hanson) Chang
"""

from tqdm import tqdm
import numpy as np

from comm.algorithms.algorithm import Algorithm

class ForwardBackward(Algorithm):
    def __init__(self, rod, algo_config, **kwargs):
        Algorithm.__init__(self, rod, algo_config)
        self.costate = Costate(self.static_rod.n_elements)
        self.stepsize = self.config.get('stepsize', 1e-8)
        self.iteration = 0
        self.done = False

        self.objects = kwargs.get('objects', kwargs.get('object', None))

    def update(self, iteration):
        return iteration+1

    def run(self, max_iter_number=100_000, **kwargs):
        print("Running the algorithm with objects:", self.objects)
        for _ in tqdm(range(max_iter_number)):
            self.iteration = self.update(self.iteration)
            if self.done:
                print("Finishing the algorithm at iternation", self.iteration)
                break
        print("Finishing the algorithm at maximum iternation", self.iteration)
        return

class Costate(object):
    def __init__(self, n_elements):
        # material frame
        self.internal_force = np.zeros((3, n_elements))
        self.internal_couple = np.zeros((3, n_elements-1))

        # lab frame
        self.internal_force_discrete_jump = np.zeros((3, n_elements))
        self.internal_couple_discrete_jump = np.zeros((3, n_elements))
        self.internal_force_derivative = np.zeros((3, n_elements))
        self.internal_couple_derivative = np.zeros((3, n_elements))
