__doc__ = """
Target Implementation
"""

import numpy as np

# from objects.object import Object

class Target:  # TODO: Maybe just use dataclass?
    def __init__(self, target_cost_weight):
        self.target_cost_weight = target_cost_weight
