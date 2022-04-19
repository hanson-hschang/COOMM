"""
Created on Sep. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""

from collections import defaultdict
import numpy as np
from tqdm import tqdm

import sys
#sys.path.append("../")          # include examples directory
#sys.path.append("../../")       # include ActuationModel directory

from set_environment import Environment
from comm.algorithms import ForwardBackwardMuscle
from comm.objects import PointTarget
from comm.callback_func import AlgorithmMuscleCallBack

def get_algo(rod, muscles, target):
    algo = ForwardBackwardMuscle(
        rod=rod,
        muscles=muscles,
        algo_config = dict(
            stepsize=1e-9,
            activation_diff_tolerance=1e-14
        ),
        object=PointTarget.get_point_target_from_sphere(
            sphere=target,
            n_elements=1,
            cost_weight=dict(
                position=0,
                director=0
            ),
            target_cost_weight=dict(
                position=1e5,
                director=1e1
            ),
            director_cost_flag=True,
        )
    )
    director = np.eye(3)
    base_to_target = algo.objects.position - rod.position_collection[:, 0]
    tip_to_target = algo.objects.position - rod.position_collection[:, -1]
    base_to_target /= np.linalg.norm(base_to_target)
    tip_to_target /= np.linalg.norm(tip_to_target)

    director[1, :] = np.cross(base_to_target, tip_to_target)
    director[0, :] = np.cross(director[1, :], tip_to_target)
    director[2, :] = np.cross(director[0, :], director[1, :])

    algo.objects.director = director.copy()
    target.director_collection[:, :, 0] = director.copy()
    return algo

def main(filename):

    """ Create simulation environment """
    final_time = 15.001
    env = Environment(final_time, recording_fps=25)
    total_steps, systems = env.reset()
    controller_Hz = 500
    controller_step_skip = int(1.0 / (controller_Hz * env.time_step))

    """ Read arm params """
    n_elements = systems[0].n_elems

    activations = []
    for m in range(len(env.muscle_groups)):
        activations.append(
            np.zeros(env.muscle_groups[m].activation.shape)
        )

    """ Initialize algorithm """
    algo = get_algo(
        rod=systems[0],
        muscles=env.muscle_groups,
        target=systems[1]
    )
    algo_callback = AlgorithmMuscleCallBack(step_skip=env.step_skip)

    algo.run(max_iter_number=50_000)
    
    
    """ Start the simulation """
    print("Running simulation ...")
    time = np.float64(0.0)
    weight_start_time = np.float64(0.0)
    for k_sim in tqdm(range(total_steps)):

        if (k_sim % controller_step_skip) == 0:
            # controller implementation
            weight = np.min([1., (time-weight_start_time)/1.])
            for m in range(len(activations)):
                activations[m] = weight*algo.activations[m]

        algo_callback.make_callback(algo, time, k_sim)
        time, systems, done = env.step(time, activations)
        if done:
            break

    """ Save the data of the simulation """
    env.save_data(
        filename=filename,
        algo=algo_callback.callback_params,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Run simulation and save result data as pickle files.'
    )
    parser.add_argument(
        '--filename', type=str, default='simulation',
        help='a str: data file name',
    )
    args = parser.parse_args()
    main(filename=args.filename)
