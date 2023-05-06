import math

import numpy as np


def nuplan_to_metadrive_vector(vector, nuplan_center=(0, 0)):
    "All vec in nuplan should be centered in (0,0) to avoid numerical explosion"
    vector = np.array(vector)
    # if len(vector.shape) == 1:
    #     vector[1] *= -1
    # else:
    #     vector[:, 1] *= -1
    vector -= np.asarray(nuplan_center)
    return vector


def compute_angular_velocity(initial_heading, final_heading, dt):
    """
    Calculate the angular velocity between two headings given in radians.

    Parameters:
    initial_heading (float): The initial heading in radians.
    final_heading (float): The final heading in radians.
    dt (float): The time interval between the two headings in seconds.

    Returns:
    float: The angular velocity in radians per second.
    """

    # Calculate the difference in headings
    delta_heading = final_heading - initial_heading

    # Adjust the delta_heading to be in the range (-π, π]
    delta_heading = (delta_heading + math.pi) % (2 * math.pi) - math.pi

    # Compute the angular velocity
    angular_vel = delta_heading / dt

    return angular_vel


def dict_recursive_remove_array(d):
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, dict):
        for k in d.keys():
            d[k] = dict_recursive_remove_array(d[k])
    return d

def mph_to_kmh(speed_in_mph: float):
    speed_in_kmh = speed_in_mph * 1.609344
    return speed_in_kmh