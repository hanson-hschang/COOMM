import numpy as np
import math

time_wave = np.sin(np.linspace(0, 100000, 1000000)) * 0.0


def draw_sucker(time_step, povray_data_folder, position, director, radius):
    with open(povray_data_folder + '/frame_sucker%04d.inc' % time_step, 'w') as file_inc:

        string = ""

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.plot(position[0, :], position[1, :], position[2, :], 'k:')
        XX_list = []

        for j in [1,-1]:
            XX_side = []
            for k in [10,20,30, 40,50, 60,65,70,75, 80,85,90,92,94,96,98]:
                Y = 0.5*radius[k] * np.sin(np.linspace(0, 380, 10) / 180 * np.pi)
                Z = 0.5*radius[k] * np.cos(np.linspace(0, 380, 10) / 180 * np.pi)
                X = np.zeros_like(Y)
                circle = [X, Z, Y]
                # ax.plot(circle[0], circle[1], circle[2], 'g')
                a = np.dot(director[:, :, k].T, circle)

                ah = (45 * j) / 180 * np.pi
                Vh = director[2, :, k]
                surface_direction = director[0, :, k] * radius[k]
                # ax.plot([position[0, k], position[0, k] + 0.01 * surface_direction[0]],
                #         [position[1, k], position[1, k] + 0.01 * surface_direction[1]],
                #         [position[2, k], position[2, k] + 0.01 * surface_direction[2]], 'r')
                # ax.plot([position[0, k], position[0, k] + 0.01 * Vh[0]],
                #         [position[1, k], position[1, k] + 0.01 * Vh[1]],
                #         [position[2, k], position[2, k] + 0.01 * Vh[2]], 'b')

                R = rotation_matrix(Vh, ah)
                XX = np.dot(R, surface_direction)
                XX = XX + position[:, k]
                # ax.plot(XX[0], XX[1], XX[2], 'ro')
                XX_circle = XX[:, np.newaxis] + np.dot(R, a)

                # ax.plot(XX_circle[0], XX_circle[1], XX_circle[2], 'go')

                XX_side.append(XX_circle.T)
            XX_list.append(XX_side)
        # ax.set_xlim(-0.0, 0.2)
        # ax.set_ylim(-0.1, 0.1)
        # ax.set_zlim(-0.1, 0.1)
        # plt.show()
        # exit()
        print(np.array(XX_list).shape)
        # exit()
        for j in [0,1]:
            sucker_radius = np.linspace(0.003, 0.0005,len(XX_list[j]))
            for body_part_index in range(len(XX_list[j])):
                body_part = XX_list[j][body_part_index]
                n_elem = len(body_part)
                string += "sphere_sweep{\n\tb_spline %d" % n_elem

                for n in range(n_elem):
                    elem = body_part[n]
                    fake_shift = time_wave[time_step] * n / n_elem * 0.025
                    string += (",\n\t<%f,%f,%f>,%f" % (elem[0] ,#+ fake_shift * np.sin(time_step / 100),
                                                       elem[1] ,#+ fake_shift * np.cos(time_step / 100),
                                                       elem[2] ,#+ fake_shift * np.sin(time_step / 100) * np.cos(n),
                                                       sucker_radius[body_part_index]))
                string += "\n\ttexture{\n"
                string += "\t\tpigment{ color Yellow transmit %f }\n" % 0.0 #rgb < 0.45, 0.39, 1 >
                string += "\t\tfinish{ phong 1 }\n\t}\n"
                string += "\tscale<16,16,16>\n}\n"

        file_inc.writelines(
            string
        )


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # xh=np.linspace(0.15,0.02,200)
    # yh=np.linspace(0,1,200)
    # zh=((-6*yh+2)**3/3+(-6*yh+2)**2+4)/11
    #
    # ax.plot(xh, yh, zh,'k')
    # ax.plot(np.zeros_like(xh), yh, zh,'r')
    #
    #
    # for j in range(1,3):
    #     for i in range(0,200,20):
    #         XX_list = []
    #         for k in range(1,365,30):
    #             ah=(60*j)/180*np.pi
    #             V=[0,yh[i+1]-yh[i],zh[i+1]-zh[i]]
    #             Vh=V/np.linalg.norm(V)
    #
    #             X=xh[i]
    #             Y=0.25*xh[i]*np.sin(k/180*np.pi)
    #             Z=0.25*xh[i]*np.cos(k/180*np.pi)
    #
    #             R=rotation_matrix(Vh,ah)
    #             XX=np.dot(R, [X,Y,Z])
    #             XX = XX+ np.array([0,yh[i], zh[i]])
    #             XX_list.append(XX)
    #         data=np.array(XX_list)
    #         ax.plot(data[:,0], data[:,1], data[:,2])

    position = np.load("position.npy")
    director = np.load("director.npy")
    radius = np.load("radius.npy")
    print(radius)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(position[0, :], position[1, :], position[2, :], 'k:')
    XX_list = []
    for j in [1, -1]:
        for k in [20]:
            Y = 0.01 * np.sin(np.linspace(0, 360, 10) / 180 * np.pi)
            Z = 0.01 * np.cos(np.linspace(0, 360, 10) / 180 * np.pi)
            X = np.zeros_like(Y)
            circle = [X, Z, Y]
            ax.plot(circle[0], circle[1], circle[2], 'g')
            a = np.dot(director[:, :, k].T, circle)

            ah = (20 * j) / 180 * np.pi
            Vh = director[2, :, k]
            surface_direction = director[0, :, k] * radius[k] * 10
            ax.plot([position[0, k], position[0, k] + 0.01 * surface_direction[0]],
                    [position[1, k], position[1, k] + 0.01 * surface_direction[1]],
                    [position[2, k], position[2, k] + 0.01 * surface_direction[2]], 'r')
            ax.plot([position[0, k], position[0, k] + 0.01 * Vh[0]],
                    [position[1, k], position[1, k] + 0.01 * Vh[1]],
                    [position[2, k], position[2, k] + 0.01 * Vh[2]], 'b')

            R = rotation_matrix(Vh, ah)
            XX = np.dot(R, surface_direction)
            XX = XX + position[:, k]
            ax.plot(XX[0], XX[1], XX[2], 'ro')
            XX_circle = XX[:, np.newaxis] + np.dot(R, a)

            ax.plot(XX_circle[0], XX_circle[1], XX_circle[2], 'go')
            XX_list.append(XX)
        data = np.array(XX_list)
        # ax.plot(data[:, 0], data[:, 1], data[:, 2])
    ax.set_xlim(-0.1, 0.25)
    ax.set_ylim(-0.125, 0.125)
    ax.set_zlim(-0.125, 0.125)
    plt.show()
    print("done")
