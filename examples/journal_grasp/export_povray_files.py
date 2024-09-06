"""
Created on Dec. 20, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import pickle
import numpy as np
from tqdm import tqdm

from coomm._rendering_tool import check_folder
from coomm.povray import (
    POVRAYFrame,
    POVRAYCamera,
)
from coomm.povray.rod import POVRAYRod
from coomm.povray.muscles import (
    POVRAYTransverseMuscle,
    POVRAYLongitudinalMuscle,
    POVRAYObliqueMuscle
)
from coomm.povray.cylinder import POVRAYCylinder


def main(filename):
    with_sucker=bool(int(input("print sucker? 0: no 1: yes")))
    if with_sucker:
        rod_alpha=1.0
    else:
        rod_alpha=0.3
    with open(filename+"_data.pickle", "rb") as f:
        data = pickle.load(f)
        rod_data = data['systems'][0]
        cylinder_data = data['systems'][1]
        muscle_groups_data = data['muscle_groups']
        
    
    povray_data_folder = filename+"_povray"
    check_folder(povray_data_folder)

    povray_camera = POVRAYCamera(
        position=[1.5, -5.0, 0.6],
        # position=[0.5, -5.0, 5.0],
        # position=[-10.0, 0.0, 0.0],
        look_at=[1.5, 0.0, 0.6],
        angle=40.0,
        floor=False
    )
    if with_sucker:
        povray_frame = POVRAYFrame(
            included_files=[
                povray_data_folder+"/camera.inc",
                povray_data_folder+"/frame0000.inc",
                povray_data_folder+"/frame_sucker0000.inc",
            ]
        )
    else:
        povray_frame = POVRAYFrame(
            included_files=[
                povray_data_folder + "/camera.inc",
                povray_data_folder + "/frame0000.inc",
            ]
        )
    povray_rod = POVRAYRod(color="<0.45, 0.39, 1.0>")
    povray_TM = POVRAYTransverseMuscle(muscle_color=np.array([0, 1, 0]))
    povray_LM = POVRAYLongitudinalMuscle(muscle_color=np.array([1, 0, 0]))
    povray_OM = POVRAYObliqueMuscle(muscle_color=np.array([0, 1, 1]))
    povray_target = POVRAYCylinder(color=np.array([1.0, 0.498039,0.0]))

    print("Exporting povray files and frames ...")
    frame_camera_name = "camera.inc"
    with open(povray_data_folder+'/'+frame_camera_name, 'w') as file_camera:
        povray_camera.write_to(file_camera)

    
    for k in tqdm(range(len(rod_data["time"]))):
        plot_flag = False
        frame_inc_name = "frame%04d.inc" % k
        frame_sucker_inc_name = "frame_sucker%04d.inc" % k
        with open(povray_data_folder+'/'+frame_inc_name, 'w') as file_inc:
            povray_rod.write_to(
                file=file_inc,
                position_data=rod_data["position"][k],
                radius_data=rod_data["radius"][k]*1.1,
                alpha=rod_alpha
            )
            if not with_sucker:
                for muscle_group_data in muscle_groups_data:
                    for muscle_data in muscle_group_data['muscles']:
                        modified_activation = (0.5*(1-np.cos(np.pi*muscle_data['activation'][k]))+0.01)/1.01
                        if 'TM' in muscle_data['muscle_info'][k]:
                            povray_TM.write_to(
                                file=file_inc,
                                position_data=rod_data["position"][k],
                                director_data=rod_data["director"][k],
                                muscle_position_data=muscle_data['muscle_position'][k],
                                radius_data=rod_data["radius"][k]*(0.0045/0.012),
                                muscle_activation=modified_activation,
                                alpha=1.0
                            )
                        if 'LM' in muscle_data['muscle_info'][k]:
                            povray_LM.write_to(
                                file=file_inc,
                                position_data=rod_data["position"][k],
                                director_data=rod_data["director"][k],
                                muscle_position_data=muscle_data['muscle_position'][k],
                                radius_data=rod_data["radius"][k]*(0.003/0.012),
                                muscle_activation=modified_activation,
                                alpha=1.0
                            )
                        if 'OM' in muscle_data['muscle_info'][k]:
                            povray_OM.write_to(
                                file=file_inc,
                                position_data=rod_data["position"][k],
                                director_data=rod_data["director"][k],
                                muscle_position_data=muscle_data['muscle_position'][k],
                                radius_data=rod_data["radius"][k]*(0.00075/0.012)*0.8,
                                muscle_activation=modified_activation,
                                alpha=1.0
                            )

            povray_target.write_to(
                file=file_inc,
                position_data=cylinder_data["position"][k],
                director_data=cylinder_data["director"][k],
                height_data=cylinder_data["height"][k],
                radius_data=cylinder_data["radius"][k],
                alpha=1.0
            )

        if with_sucker:
            draw_sucker(k,povray_data_folder,rod_data["position"][k],rod_data["director"][k],rod_data["radius"][k])

        frame_povray_name = "frame%04d.pov" % k
        povray_frame.included_files[1] = povray_data_folder+'/'+frame_inc_name
        if with_sucker:
            povray_frame.included_files[2] = povray_data_folder+'/'+frame_sucker_inc_name
        with open(povray_data_folder+'/'+frame_povray_name, 'w') as file_frame:
            povray_frame.write_included_files_to(file_frame)

        povray_frame_file_name = povray_data_folder+'/'+frame_povray_name
        import subprocess
        # subprocess.run(["povray", "-H400", "-W600", "Quality=11", "Antialias=on", povray_frame_file_name])
        subprocess.run(["povray", "-H1080", "-W1080", "Quality=11", "Antialias=on", povray_frame_file_name])
        # subprocess.run(["povray", "-H2100", "-W3150", "Quality=11", "Antialias=on", povray_frame_file_name])
        # subprocess.run(["povray", "-H480", "-W480", "Quality=11", "Antialias=on", povray_frame_file_name])
        # subprocess.run(["povray", "-H960", "-W960", "Quality=11", "Antialias=on", povray_frame_file_name])
        # subprocess.run(["povray", "-H1920", "-W1920", "Quality=11", "Antialias=on", povray_frame_file_name])
        # subprocess.run(["povray", "-H3840", "-W3840", "Quality=11", "Antialias=on", povray_frame_file_name])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Export simulation result as povray files and frames in filename_povray.'
    )
    parser.add_argument(
        '--filename', type=str, default='simulation',
        help='a str: data file name',
    )
    args = parser.parse_args()
    main(filename=args.filename)
