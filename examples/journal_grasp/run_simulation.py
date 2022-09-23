"""
Created on Sep. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from tqdm import tqdm

import sys
sys.path.append("../")          # include examples directory
# sys.path.append("../../")       # include coomm directory

from coomm.objects import CylinderTarget
from coomm.algorithms import ForwardBackwardMuscle
from coomm.callback_func import AlgorithmMuscleCallBack

from set_environment import Environment
from plot_frames import Frame

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(-height_z, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

def get_algo(rod, muscles, target):
    s = np.linspace(0, 1, rod.n_elems)
    weight = np.zeros(s.shape)
    weight = 0.5*(1+np.tanh((s-0.3)*100))
    
    algo = ForwardBackwardMuscle(
        rod=rod,
        muscles=muscles,
        algo_config=dict(
            stepsize=1e-8,
            activation_diff_tolerance=1e-12
        ),
        object=CylinderTarget.get_cylinder(
            cylinder=target,
            n_elements=100,
            cost_weight=dict(
                position=1e7*np.ones(s.shape),
                director=0
            ),
            target_cost_weight=dict(
                position=1e6*weight,
                director=1e3*weight
            ),
        )
    )
    # director = np.eye(3)
    # base_to_target = algo.objects.position - rod.position_collection[:, 0]
    # tip_to_target = algo.objects.position - rod.position_collection[:, -1]
    # base_to_target /= np.linalg.norm(base_to_target)
    # tip_to_target /= np.linalg.norm(tip_to_target)

    # director[1, :] = np.cross(base_to_target, tip_to_target)
    # director[0, :] = np.cross(director[1, :], tip_to_target)
    # director[2, :] = np.cross(director[0, :], director[1, :])

    # algo.objects.director = director.copy()
    # target.director_collection[:, :, 0] = director.copy()
    return algo

def main(filename, target_position=None, target_director=None):

    """ Create simulation environment """
    final_time = 15.001
    env = Environment(final_time)
    total_steps, systems = env.reset()
    controller_Hz = 500
    controller_step_skip = int(1.0 / (controller_Hz * env.time_step))

    """ Initialize algorithm """
    algo = get_algo(
        rod=systems[0], 
        muscles=env.muscle_groups,
        target=systems[1]
    )
    algo_callback = AlgorithmMuscleCallBack(step_skip=env.step_skip)


    # frame = Frame.get_frame(filename)
    # L0 = frame.set_ref_configuration(
    #     position=systems[0].position_collection,
    #     shear=systems[0].sigma+np.array([0, 0, 1])[:, None],
    #     kappa=systems[0].kappa,
    # )
    # frame.reset()

    # ax_main = frame.plot_rod(
    #     position=algo.static_rod.position_collection,
    #     director=algo.static_rod.director_collection,
    #     radius=algo.static_rod.radius,
    #     color='orange',
    #     alpha=0.1
    # )

    # base = systems[1].position_collection[:, 0]/L0
    # ax_main.scatter(
    #     base[0],
    #     base[1],
    #     base[2],
    #     color='grey'
    # )
    # cylinder = systems[1]
    # Xc,Yc,Zc = data_for_cylinder_along_z(
    #     cylinder.position_collection[0, 0]/L0,
    #     cylinder.position_collection[1, 0]/L0,
    #     cylinder.radius/L0,
    #     cylinder.length/L0/2
    # )
    # ax_main.plot_surface(Xc, Yc, Zc, alpha=0.5)

    # n_ii = 2
    # for ii in range(n_ii):
    #     algo.run(max_iter_number=50_000)
    #     frame.plot_rod(
    #         position=algo.static_rod.position_collection,
    #         director=algo.static_rod.director_collection,
    #         radius=algo.static_rod.radius,
    #         color='orange',
    #         alpha=0.1+(ii+1)*(0.9/n_ii)
    #     )

    #     for i in range(3):
    #         director_line = np.zeros((3, 2))
    #         director_line[:, 0] = base.copy()
    #         director_line[:, 1] = base + systems[1].director_collection[i, :, 0] * 0.1
    #         ax_main.plot(
    #             director_line[0], director_line[1], director_line[2],
    #             color='grey',
    #         )
        
    #     for n in [19, 39, 59, 79, 99]:
    #         base = algo.static_rod.position_collection[:, n]/L0
    #         ax_main.scatter(
    #             base[0],
    #             base[1],
    #             base[2],
    #             color='red'
    #         )
    #         for i in range(3):
    #             director_line = np.zeros((3, 2))
    #             director_line[:, 0] = base.copy()
    #             director_line[:, 1] = base + algo.static_rod.director_collection[i, :, n] * 0.1
    #             color = 'green' if i==0 else 'red'
    #             ax_main.plot(
    #                 director_line[0], director_line[1], director_line[2],
    #                 color=color,
    #             )

    #     frame.set_ax_main_lim(
    #         x_lim=[-1.1, 1.1],
    #         y_lim=[-1.1, 1.1],
    #         z_lim=[-1.1, 1.1]
    #     )
        
    #     frame.plot_strain(
    #         shear=algo.static_rod.sigma+np.array([0, 0, 1])[:, None],
    #         kappa=algo.static_rod.kappa
    #     )
        
    #     for activation in algo.activations:
    #         print(max(activation))
    # frame.show()

    # quit()

    """ Run the algorithm """
    # algo.run(max_iter_number=200_000)
    algo.run(max_iter_number=100_000)
    # algo.run(max_iter_number=10_000)
    for activation in algo.activations:
        print(max(activation))

    activations = []
    for m in range(len(env.muscle_groups)):
        activations.append(
            np.zeros(env.muscle_groups[m].activation.shape)
        )

    """ Start the simulation """
    print("Running simulation ...")
    time = np.float64(0.0)
    for k_sim in tqdm(range(total_steps)):

        if (k_sim % controller_step_skip) == 0:
            # controller implementation
            weight = np.min([1., time/1.])
            for m in range(len(activations)):
                activations[m] = weight*algo.activations[m]
     
        algo_callback.make_callback(algo, time, k_sim)
        time, systems, done = env.step(time, activations)
        if done:
            print("Exiting simulation with error(s) occured ...")
            break
    if not done:
        print("Simulation completed ...")
        
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
