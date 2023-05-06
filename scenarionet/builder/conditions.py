import numpy as np


def validate_sdc_track(sdc_state):
    """
    This function filters the scenario based on SDC information.

    Rule 1: Filter out if the trajectory length < 10

    Rule 2: Filter out if the whole trajectory last < 5s, assuming sampling frequency = 10Hz.
    """
    valid_array = sdc_state["valid"]
    sdc_trajectory = sdc_state["position"][valid_array, :2]
    sdc_track_length = [
        np.linalg.norm(sdc_trajectory[i] - sdc_trajectory[i + 1]) for i in range(sdc_trajectory.shape[0] - 1)
    ]
    sdc_track_length = sum(sdc_track_length)

    # Rule 1
    if sdc_track_length < 10:
        return False

    print("sdc_track_length: ", sdc_track_length)

    # Rule 2
    if valid_array.sum() < 50:
        return False

    return True
