import ast
import copy
import inspect
import math
import os
import pickle
import shutil
from collections import defaultdict

import numpy as np
import tqdm
from metadrive.scenario import ScenarioDescription as SD, ScenarioDescription


def nuplan_to_metadrive_vector(vector, nuplan_center=(0, 0)):
    "All vec in nuplan should be centered in (0,0) to avoid numerical explosion"
    vector = np.array(vector)
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


def get_agent_summary(state_dict, id, type):
    track = state_dict["position"]
    valid_track = track[state_dict["valid"].astype(int), :2]
    distance = float(sum(np.linalg.norm(valid_track[i] - valid_track[i + 1]) for i in range(valid_track.shape[0] - 1)))
    valid_length = int(sum(state_dict["valid"]))

    continuous_valid_length = 0
    for v in state_dict["valid"]:
        if v:
            continuous_valid_length += 1
        if continuous_valid_length > 0 and not v:
            break

    return {
        "type": type,
        "object_id": str(id),
        "track_length": int(len(track)),
        "distance": float(distance),
        "valid_length": int(valid_length),
        "continuous_valid_length": int(continuous_valid_length)
    }


def get_number_summary(scenario):
    number_summary_dict = {}
    number_summary_dict["object"] = len(scenario[SD.TRACKS])
    number_summary_dict["dynamic_object_states"] = len(scenario[SD.DYNAMIC_MAP_STATES])
    number_summary_dict["map_features"] = len(scenario[SD.MAP_FEATURES])
    number_summary_dict["object_types"] = set(v["type"] for v in scenario[SD.TRACKS].values())

    object_types_counter = defaultdict(int)
    for v in scenario[SD.TRACKS].values():
        object_types_counter[v["type"]] += 1
    number_summary_dict["object_types_counter"] = dict(object_types_counter)

    # Number of different dynamic object states
    dynamic_object_states_types = set()
    dynamic_object_states_counter = defaultdict(int)
    for v in scenario[SD.DYNAMIC_MAP_STATES].values():
        for step_state in v["state"]["object_state"]:
            if step_state is None:
                continue
            dynamic_object_states_types.add(step_state)
            dynamic_object_states_counter[step_state] += 1
    number_summary_dict["dynamic_object_states_types"] = dynamic_object_states_types
    number_summary_dict["dynamic_object_states_counter"] = dict(dynamic_object_states_counter)

    return number_summary_dict


def contains_explicit_return(f):
    return any(isinstance(node, ast.Return) for node in ast.walk(ast.parse(inspect.getsource(f))))


def write_to_directory(convert_func,
                       scenarios,
                       output_path,
                       dataset_version,
                       dataset_name,
                       force_overwrite=False,
                       **kwargs):
    """
    Convert a batch of scenarios.
    """
    if not contains_explicit_return(convert_func):
        raise RuntimeError("The convert function should return a metadata dict")

    save_path = copy.deepcopy(output_path)
    output_path = output_path + "_tmp"
    # meta recorder and data summary
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=False)

    # make real save dir
    delay_remove = None
    if os.path.exists(save_path):
        if force_overwrite:
            delay_remove = save_path
        else:
            raise ValueError("Directory already exists! Abort")

    summary_file = "dataset_summary.pkl"

    metadata_recorder = {}
    for scenario in tqdm.tqdm(scenarios):
        # convert scenario
        sd_scenario, scenario_id = convert_func(scenario, **kwargs)
        export_file_name = "sd_{}_{}.pkl".format(dataset_name + "_" + dataset_version, scenario_id)

        # add agents summary
        summary_dict = {}
        ego_car_id = sd_scenario[SD.METADATA][SD.SDC_ID]
        summary_dict[ego_car_id] = get_agent_summary(
            state_dict=sd_scenario.get_sdc_track()["state"], id=ego_car_id, type=sd_scenario.get_sdc_track()["type"]
        )
        for track_id, track in sd_scenario[SD.TRACKS].items():
            summary_dict[track_id] = get_agent_summary(state_dict=track["state"], id=track_id, type=track["type"])
        sd_scenario[SD.METADATA]["object_summary"] = summary_dict

        # count some objects occurrence
        sd_scenario[SD.METADATA]["number_summary"] = get_number_summary(sd_scenario)
        metadata_recorder[export_file_name] = copy.deepcopy(sd_scenario[SD.METADATA])

        # sanity check
        sd_scenario = sd_scenario.to_dict()
        ScenarioDescription.sanity_check(sd_scenario, check_self_type=True)

        # dump
        p = os.path.join(output_path, export_file_name)
        with open(p, "wb") as f:
            pickle.dump(sd_scenario, f)

    # rename and save
    if delay_remove is not None:
        assert delay_remove == save_path
        shutil.rmtree(delay_remove)
    os.rename(output_path, save_path)
    summary_file = os.path.join(save_path, summary_file)
    with open(summary_file, "wb") as file:
        pickle.dump(dict_recursive_remove_array(metadata_recorder), file)
    print("Summary is saved at: {}".format(summary_file))

