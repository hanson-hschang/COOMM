__doc__ = """
Base object module
"""

import numpy as np

class Object:  # FIXME: To general name.
    """Object.
    """

    def __init__(self, n_elements, cost_weight=None):
        """__init__.

        Parameters
        ----------
        n_elements :
            n_elements
        cost_weight :
            cost_weight
        """
        self.n_elements = n_elements
        self.cost = Cost(n_elements)
        self.cost_gradient = CostGradient(n_elements)
        self.cost_weight = cost_weight

    def __call__(self, **kwargs): # FIXME: Maybe avoid using __call__?
        """__call__
        """
        # All cost related terms should be calculated once this function is called
        # including running cost, terminal cost (might not just on terminal it should 
        # be extend to any or some specific s as well), running cost gradient and 
        # terminal cost gradient (similarly, should be extended to any or some specific s)
        self.calculate_cost_gradient(**kwargs)

    # def calculate_cost(self, **kwargs):
    #     # should return all cost terms including running cost and terminal cost
    #     return NotImplementedError

    def calculate_cost_gradient(self, **kwargs):
        """calculate_cost_gradient.
        """
        self.calculate_continuous_cost_gradient_wrt_position(**kwargs)
        self.calculate_continuous_cost_gradient_wrt_director(**kwargs)
        self.calculate_discrete_cost_gradient_wrt_position(**kwargs)
        self.calculate_discrete_cost_gradient_wrt_director(**kwargs)

    def calculate_continuous_cost_gradient_wrt_position(self, **kwargs):
        return NotImplementedError
    
    def calculate_continuous_cost_gradient_wrt_director(self, **kwargs):
        return NotImplementedError
    
    def calculate_discrete_cost_gradient_wrt_position(self, **kwargs):
        return NotImplementedError
    
    def calculate_discrete_cost_gradient_wrt_director(self, **kwargs):
        return NotImplementedError

class Cost:
    """Cost.
    """

    def __init__(self, n_elements):
        """__init__.

        Parameters
        ----------
        n_elements :
            n_elements
        """
        self.continuous = WRT_Pose(n_elements, n_elements, dim=1)
        self.discrete = WRT_Pose(n_elements, n_elements, dim=1)

    def reset(self,):
        """reset.
        """
        self.continuous.reset()
        self.discrete.reset()

class CostGradient():
    """CostGradient.
    """

    def __init__(self, n_elements):
        """__init__.

        Parameters
        ----------
        n_elements :
            n_elements
        """
        self.continuous = WRT_Pose(n_elements, n_elements)
        self.discrete = WRT_Pose(n_elements, n_elements)

    def reset(self,):
        """reset.
        """
        self.continuous.reset()
        self.discrete.reset()

    def add(self, other):
        """add.

        Parameters
        ----------
        other :
            other
        """
        self.continuous.add(other.continuous)
        self.discrete.add(other.discrete)

class WRT_Pose():
    """WRT_Pose.
    """

    def __init__(self, n_elements_for_position, n_elements_for_director, dim=3):
        """__init__.

        Parameters
        ----------
        n_elements_for_position :
            n_elements_for_position
        n_elements_for_director :
            n_elements_for_director
        dim :
            dim
        """
        if dim ==1:
            self.wrt_position = np.zeros(n_elements_for_position)
            self.wrt_director = np.zeros(n_elements_for_director)
        else:
            self.wrt_position = np.zeros((dim, n_elements_for_position))
            self.wrt_director = np.zeros((dim, n_elements_for_director))
    
    def reset(self,):
        """reset.
        """
        self.wrt_position *= 0
        self.wrt_director *= 0
    
    def add(self, other):
        """add.

        Parameters
        ----------
        other :
            other
        """
        self.wrt_position += other.wrt_position
        self.wrt_director += other.wrt_director

class Objects(Object): # FIXME: we should be clear on naming.
    """Objects.
    """

    def __init__(self, objects):
        """__init__.

        Parameters
        ----------
        objects :
            objects
        """
        Object.__init__(self, objects[0].n_elements)
        self.objects = objects
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return Objects(
                [self.objects[k] 
                for k in range(*key.indices(len(self.objects)))
                ]
            )
        return self.objects[key]
    
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            for k, v in zip(
                range(*key.indices(len(self.objects))), value
            ):
                self.objects[k] = v
        else:
            self.objects[key] = value
    
    def append(self, value):
        """append.

        Parameters
        ----------
        value :
            value
        """
        self.objects.append(value)

    def __call__(self, **kwargs):
        self.reset()
        for obj in self.objects:
            obj(**kwargs)
            self.cost_gradient.add(obj.cost_gradient)

    def calculate_cost_gradient(self, **kwargs):
        """calculate_cost_gradient.
        """
        pass

    def calculate_continuous_cost_gradient_wrt_position(self, **kwargs):
        """calculate_continuous_cost_gradient_wrt_position.
        """
        pass
    
    def calculate_continuous_cost_gradient_wrt_director(self, **kwargs):
        """calculate_continuous_cost_gradient_wrt_director.
        """
        pass
    
    def calculate_discrete_cost_gradient_wrt_position(self, **kwargs):
        """calculate_discrete_cost_gradient_wrt_position.
        """
        pass
    
    def calculate_discrete_cost_gradient_wrt_director(self, **kwargs):
        """calculate_discrete_cost_gradient_wrt_director.
        """
        pass
