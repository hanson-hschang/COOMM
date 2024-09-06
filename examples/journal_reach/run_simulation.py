"""
Created on Sep. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from tqdm import tqdm

from set_environment import Environment

from coomm.algorithms import ForwardBackwardMuscle
from coomm.objects import PointTarget
from coomm.callback_func import AlgorithmMuscleCallBack

def get_algo(rod, muscles, target):
    algo = ForwardBackwardMuscle(
        rod=rod,
        muscles=muscles,
        algo_config = dict(
            stepsize=1e-8,
            activation_diff_tolerance=1e-12
        ),
        object=PointTarget.get_point_target_from_sphere(
            sphere=target,
            n_elements=1,
            cost_weight=dict(
                position=0,
                director=0
            ),
            target_cost_weight=dict(
                position=1e6,
                director=1e3
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

def main(filename, target_position=None):

    """ Create simulation environment """
    final_time = 15.001
    env = Environment(final_time)
    total_steps, systems = env.reset()
    controller_Hz = 500
    controller_step_skip = int(1.0 / (controller_Hz * env.time_step))

    if not (target_position is None):
        env.sphere.position_collection[:, 0] = target_position

    """ Initialize algorithm """
    algo = get_algo(
        rod=systems[0],
        muscles=env.muscle_groups,
        target=systems[1]
    )
    algo_callback = AlgorithmMuscleCallBack(step_skip=env.step_skip)

    algo.run(max_iter_number=100_000)
    
    """ Read arm params """
    activations = []
    for m in range(len(env.muscle_groups)):
        activations.append(
            np.zeros(env.muscle_groups[m].activation.shape)
        )

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
