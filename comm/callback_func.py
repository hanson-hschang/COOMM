__doc__ = """
Common callback function used in COMM.
"""

from collections import defaultdict

from elastica.wrappers import callbacks

from elastica.callback_functions import CallBackBaseClass

class BasicCallBackBaseClass(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.save_params(system, time)
    
    def save_params(self, system, time):
        return NotImplementedError

class RodCallBack(BasicCallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        BasicCallBackBaseClass.__init__(self, step_skip, callback_params)

    def save_params(self, system, time):
        self.callback_params["time"].append(time)
        self.callback_params["radius"].append(system.radius.copy())
        self.callback_params["dilatation"].append(system.dilatation.copy())
        self.callback_params["voronoi_dilatation"].append(system.voronoi_dilatation.copy())
        self.callback_params["position"].append(system.position_collection.copy())
        self.callback_params["director"].append(system.director_collection.copy())
        self.callback_params["velocity"].append(system.velocity_collection.copy())
        self.callback_params["omega"].append(system.omega_collection.copy())
        self.callback_params["sigma"].append(system.sigma.copy())
        self.callback_params["kappa"].append(system.kappa.copy())

class ExternalLoadCallBack(BasicCallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        BasicCallBackBaseClass.__init__(self, step_skip, callback_params)

    def save_params(self, system, time):
        self.callback_params["time"].append(time)
        self.callback_params['external_force'].append(system.external_forces.copy())
        self.callback_params['external_couple'].append(system.external_torques.copy())

class CylinderCallBack(BasicCallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        BasicCallBackBaseClass.__init__(self, step_skip, callback_params)

    def save_params(self, system, time):
        self.callback_params["time"].append(time)
        self.callback_params["radius"].append(system.radius)
        self.callback_params["height"].append(system.length)
        self.callback_params["position"].append(system.position_collection.copy())
        self.callback_params['director'].append(system.director_collection.copy())

class SphereCallBack(BasicCallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        BasicCallBackBaseClass.__init__(self, step_skip, callback_params)

    def save_params(self, system, time):
        self.callback_params["time"].append(time)
        self.callback_params["radius"].append(system.radius)
        self.callback_params["position"].append(system.position_collection.copy())
        self.callback_params["director"].append(system.director_collection.copy())

class AlgorithmCallBack(BasicCallBackBaseClass):
    def __init__(self, step_skip: int):
        callback_params = defaultdict(list)
        BasicCallBackBaseClass.__init__(self, step_skip, callback_params)

    def save_params(self, system, time):
        self.callback_params["time"].append(time)
        self.callback_params["radius"].append(system.static_rod.radius.copy())
        self.callback_params["dilatation"].append(system.static_rod.dilatation.copy())
        self.callback_params["voronoi_dilatation"].append(system.static_rod.voronoi_dilatation.copy())
        self.callback_params["position"].append(system.static_rod.position_collection.copy())
        self.callback_params["director"].append(system.static_rod.director_collection.copy())
        self.callback_params["sigma"].append(system.static_rod.sigma.copy())
        self.callback_params["kappa"].append(system.static_rod.kappa.copy())

class AlgorithmMuscleCallBack(AlgorithmCallBack):
    def __init__(self, step_skip: int):
        AlgorithmCallBack.__init__(self, step_skip)

    def save_params(self, system, time):
        AlgorithmCallBack.save_params(self, system, time)
        self.callback_params["activations"].append(system.activations.copy())

class OthersCallBack:
    def __init__(self, step_skip: int, callback_params: dict):
        self.current_step = 0
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, time, **kwargs):
        if self.current_step % self.every == 0:
            self.callback_func(time, **kwargs)
        self.current_step += 1

    def callback_func(self, time, **kwargs):
        self.callback_params['time'].append(time)
        for key, value in kwargs.items():
            # make sure value is a numpy array
            self.callback_params[key].append(value.copy())

    def save_data(self, **kwargs):

        import pickle

        print("Saving additional data to simulation_others.pickle file...", end='\r')

        with open("simulation_others.pickle", "wb") as others_file:
            data = dict(
                time_series_data=self.callback_params,
                **kwargs
            )
            pickle.dump(data, others_file)

        print("Saving additional data to simulation_others.pickle file... Done!")
