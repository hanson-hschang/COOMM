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
from plot_frames import Frame

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

def main(filename):

    """ Create simulation environment """
    final_time = 15.001
    env = Environment(final_time)
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
    
    frame = Frame.get_frame(filename)
    L0 = frame.set_ref_configuration(
        position=systems[0].position_collection,
        shear=systems[0].sigma+np.array([0, 0, 1])[:, None],
        kappa=systems[0].kappa,
    )
    frame.reset()

    ax_main = frame.plot_rod(
        position=algo.static_rod.position_collection,
        director=algo.static_rod.director_collection,
        radius=algo.static_rod.radius,
        color='orange',
        alpha=0.3
    )

    base = systems[1].position_collection[:, 0]/L0
    ax_main.scatter(
        base[0],
        base[1],
        base[2],
        color='grey'
    )

    for i in range(3):
        director_line = np.zeros((3, 2))
        director_line[:, 0] = base.copy()
        director_line[:, 1] = base + systems[1].director_collection[i, :, 0] * 0.1
        ax_main.plot(
            director_line[0], director_line[1], director_line[2],
            color='grey',
        )
    
    base = algo.static_rod.position_collection[:, -1]/L0
    ax_main.scatter(
        base[0],
        base[1],
        base[2],
        color='red'
    )
    for i in range(3):
        director_line = np.zeros((3, 2))
        director_line[:, 0] = base.copy()
        director_line[:, 1] = base + algo.static_rod.director_collection[i, :, -1] * 0.1
        color = 'green' if i==0 else 'red'
        ax_main.plot(
            director_line[0], director_line[1], director_line[2],
            color=color,
        )

    frame.set_ax_main_lim(
        x_lim=[-1.1, 1.1],
        y_lim=[-1.1, 1.1],
        z_lim=[-1.1, 1.1]
    )
    
    frame.plot_strain(
        shear=algo.static_rod.sigma+np.array([0, 0, 1])[:, None],
        kappa=algo.static_rod.kappa
    )

    # frame.show()
    
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
