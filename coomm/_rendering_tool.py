__doc__ = """
Processing scripts for the rendering
"""

import os, shutil
import numpy as np
from numba import njit

@njit(cache=True)
def process_position(position, offset, rotation):
    blocksize = position.shape[1]
    output_position = np.zeros((3, blocksize))
    for n in range(blocksize):
        for i in range(3):
            for j in range(3):
                output_position[i, n] += (
                    rotation[i, j] * (position[j, n] - offset[j])
                )
    return output_position

@njit(cache=True)
def process_director(director, rotation):
    blocksize = director.shape[2]
    output_director = np.zeros((3, 3, blocksize))
    for n in range(blocksize):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    output_director[i, j, n] += (
                        rotation[i, k] * director[k, j, n]
                    )
    return output_director

def check_folder(folder_name):
    if not (folder_name is None):
        if os.path.exists(folder_name):
            print('Clean up files in: {}/'.format(folder_name))
            shutil.rmtree(folder_name)
        print('Create the directory: {}/'.format(folder_name))
        os.mkdir(folder_name)
