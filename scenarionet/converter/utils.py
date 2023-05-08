import ast
import copy
import inspect
import logging
import math
import multiprocessing
import os
import pickle
import shutil
from functools import partial

import numpy as np
import tqdm
from metadrive.scenario import ScenarioDescription as SD

from scenarionet.common_utils import save_summary_anda_mapping

logger = logging.getLogger(__file__)


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


def mph_to_kmh(speed_in_mph: float):
    speed_in_kmh = speed_in_mph * 1.609344
    return speed_in_kmh


def contains_explicit_return(f):
    return any(isinstance(node, ast.Return) for node in ast.walk(ast.parse(inspect.getsource(f))))


def write_to_directory(convert_func,
                       scenarios,
                       output_path,
                       dataset_version,
                       dataset_name,
                       force_overwrite=False,
                       num_workers=8,
                       **kwargs):
    # make sure dir not exist
    basename = os.path.basename(output_path)
    dir = os.path.dirname(output_path)
    for i in range(num_workers):
        output_path = os.path.join(dir, "{}_{}".format(basename, str(i)))
        if os.path.exists(output_path):
            if not force_overwrite:
                raise ValueError("Directory {} already exists! Abort. "
                                 "\n Try setting force_overwrite=True or adding --overwrite".format(output_path))
    # get arguments for workers
    num_files = len(scenarios)
    if num_files < num_workers:
        # single process
        logger.info("Use one worker, as num_scenario < num_workers:")
        num_workers = 1

    argument_list = []
    num_files_each_worker = int(num_files // num_workers)
    for i in range(num_workers):
        if i == num_workers - 1:
            end_idx = num_files
        else:
            end_idx = (i + 1) * num_files_each_worker
        output_path = os.path.join(dir, "{}_{}".format(basename, str(i)))
        argument_list.append([scenarios[i * num_files_each_worker:end_idx], kwargs, i, output_path])

    # prefill arguments
    func = partial(writing_to_directory_wrapper,
                   convert_func=convert_func,
                   dataset_version=dataset_version,
                   dataset_name=dataset_name,
                   force_overwrite=force_overwrite)

    # Run, workers and process result from worker
    with multiprocessing.Pool(num_workers) as p:
        all_result = list(p.imap(func, argument_list))
    return all_result


def writing_to_directory_wrapper(args,
                                 convert_func,
                                 dataset_version,
                                 dataset_name,
                                 force_overwrite=False):
    return write_to_directory_single_worker(convert_func=convert_func,
                                            scenarios=args[0],
                                            output_path=args[3],
                                            dataset_version=dataset_version,
                                            dataset_name=dataset_name,
                                            force_overwrite=force_overwrite,
                                            worker_index=args[2],
                                            **args[1])


def write_to_directory_single_worker(convert_func,
                                     scenarios,
                                     output_path,
                                     dataset_version,
                                     dataset_name,
                                     worker_index=0,
                                     force_overwrite=False, **kwargs):
    """
    Convert a batch of scenarios.
    """
    if not contains_explicit_return(convert_func):
        raise RuntimeError("The convert function should return a metadata dict")

    if "version" in kwargs:
        kwargs.pop("version")
        logger.info("the specified version in kwargs is replaced by argument: 'dataset_version'")

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
            raise ValueError("Directory already exists! Abort."
                             "\n Try setting force_overwrite=True or using --overwrite")

    summary_file = SD.DATASET.SUMMARY_FILE
    mapping_file = SD.DATASET.MAPPING_FILE

    summary_file_path = os.path.join(output_path, summary_file)
    mapping_file_path = os.path.join(output_path, mapping_file)

    summary = {}
    mapping = {}
    for scenario in tqdm.tqdm(scenarios, desc="Worker Index: {}".format(worker_index)):
        # convert scenario
        sd_scenario = convert_func(scenario, dataset_version, **kwargs)
        scenario_id = sd_scenario[SD.ID]
        export_file_name = SD.get_export_file_name(dataset_name, dataset_version, scenario_id)

        # add agents summary
        summary_dict = {}
        ego_car_id = sd_scenario[SD.METADATA][SD.SDC_ID]
        summary_dict[ego_car_id] = SD.get_object_summary(
            state_dict=sd_scenario.get_sdc_track()["state"], id=ego_car_id, type=sd_scenario.get_sdc_track()["type"]
        )
        for track_id, track in sd_scenario[SD.TRACKS].items():
            summary_dict[track_id] = SD.get_object_summary(state_dict=track["state"], id=track_id, type=track["type"])
        sd_scenario[SD.METADATA][SD.SUMMARY.OBJECT_SUMMARY] = summary_dict

        # count some objects occurrence
        sd_scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY] = SD.get_number_summary(sd_scenario)

        # update summary/mapping dicy
        summary[export_file_name] = copy.deepcopy(sd_scenario[SD.METADATA])
        mapping[export_file_name] = ""  # in the same dir

        # sanity check
        sd_scenario = sd_scenario.to_dict()
        SD.sanity_check(sd_scenario, check_self_type=True)

        # dump
        p = os.path.join(output_path, export_file_name)
        with open(p, "wb") as f:
            pickle.dump(sd_scenario, f)

    # store summary file
    save_summary_anda_mapping(summary_file_path, mapping_file_path, summary, mapping)

    # rename and save
    if delay_remove is not None:
        assert delay_remove == save_path
        shutil.rmtree(delay_remove)
    os.rename(output_path, save_path)

    return summary, mapping
