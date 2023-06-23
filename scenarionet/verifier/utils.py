import logging
import multiprocessing
import os

import numpy as np

from scenarionet.common_utils import read_scenario, read_dataset_summary
from scenarionet.verifier.error import ErrorDescription as ED
from scenarionet.verifier.error import ErrorFile as EF

logger = logging.getLogger(__name__)
import tqdm
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import get_number_of_scenarios
from functools import partial

# this global variable is for generating broken scenarios for testing
RANDOM_DROP = False


def set_random_drop(drop):
    global RANDOM_DROP
    RANDOM_DROP = drop


def verify_database(dataset_path, error_file_path, overwrite=False, num_workers=8, steps_to_run=1000):
    global RANDOM_DROP
    assert os.path.isdir(error_file_path), "error_file_path must be a dir, get {}".format(error_file_path)
    os.makedirs(error_file_path, exist_ok=True)
    error_file_name = EF.get_error_file_name(dataset_path)
    if os.path.exists(os.path.join(error_file_path, error_file_name)) and not overwrite:
        raise FileExistsError(
            "An error_file already exists in result_save_directory. "
            "Setting overwrite=True to cancel this alert"
        )
    num_scenario = get_number_of_scenarios(dataset_path)
    if num_scenario < num_workers:
        # single process
        logger.info("Use one worker, as num_scenario < num_workers:")
        num_workers = 1

    # prepare arguments
    argument_list = []
    func = partial(loading_wrapper, dataset_path=dataset_path, steps_to_run=steps_to_run, random_drop=RANDOM_DROP)

    num_scenario_each_worker = int(num_scenario // num_workers)
    for i in range(num_workers):
        if i == num_workers - 1:
            scenario_num = num_scenario - num_scenario_each_worker * (num_workers - 1)
        else:
            scenario_num = num_scenario_each_worker
        argument_list.append([i * num_scenario_each_worker, scenario_num])

    # Run, workers and process result from worker
    with multiprocessing.Pool(num_workers) as p:
        all_result = list(p.imap(func, argument_list))
    success = all([i[0] for i in all_result])
    errors = []
    for _, error in all_result:
        errors += error
    # logging
    if success:
        logger.info("All scenarios can be loaded successfully!")
    else:
        # save result
        path = EF.dump(error_file_path, errors, dataset_path)
        logger.info(
            "Fail to load all scenarios. Number of failed scenarios: {}. "
            "See: {} more details! ".format(len(errors), path)
        )
    return success, errors


def loading_into_metadrive(
    start_scenario_index, num_scenario, dataset_path, steps_to_run, metadrive_config=None, random_drop=False
):
    logger.info(
        "================ Begin Scenario Loading Verification for scenario {}-{} ================ \n".format(
            start_scenario_index, num_scenario + start_scenario_index
        )
    )
    success = True
    error_msgs = []

    if steps_to_run == 0:
        summary, scenarios, mapping = read_dataset_summary(dataset_path)
        index_count = -1
        for file_name in tqdm.tqdm(scenarios):
            index_count += 1
            try:
                scenario = read_scenario(dataset_path, mapping, file_name)
                SD.sanity_check(scenario)
                if random_drop and np.random.rand() < 0.5:
                    raise ValueError("Random Drop")
            except Exception as e:
                file_path = os.path.join(dataset_path, mapping[file_name], file_name)
                error_msg = ED.make(index_count, file_path, file_name, str(e))
                error_msgs.append(error_msg)
                success = False
                # proceed to next scenario
                continue
    else:
        metadrive_config = metadrive_config or {}
        metadrive_config.update(
            {
                "agent_policy": ReplayEgoCarPolicy,
                "num_scenarios": num_scenario,
                "horizon": 1000,
                "store_map": False,
                "start_scenario_index": start_scenario_index,
                "no_static_vehicles": False,
                "data_directory": dataset_path,
            }
        )
        env = ScenarioEnv(metadrive_config)
        logging.disable(logging.INFO)
        desc = "Scenarios: {}-{}".format(start_scenario_index, start_scenario_index + num_scenario)
        for scenario_index in tqdm.tqdm(range(start_scenario_index, start_scenario_index + num_scenario), desc=desc):
            try:
                env.reset(seed=scenario_index)
                arrive = False
                if random_drop and np.random.rand() < 0.5:
                    raise ValueError("Random Drop")
                for _ in range(steps_to_run):
                    o, r, d, _, info = env.step([0, 0])
                    if d and info["arrive_dest"]:
                        arrive = True
                assert arrive, "Can not arrive destination"
            except Exception as e:
                file_name = env.engine.data_manager.summary_lookup[scenario_index]
                file_path = os.path.join(dataset_path, env.engine.data_manager.mapping[file_name], file_name)
                error_msg = ED.make(scenario_index, file_path, file_name, str(e))
                error_msgs.append(error_msg)
                success = False
                # proceed to next scenario
                continue

        env.close()
    return success, error_msgs


def loading_wrapper(arglist, dataset_path, steps_to_run, random_drop):
    assert len(arglist) == 2, "Too much arguments!"
    return loading_into_metadrive(
        arglist[0], arglist[1], dataset_path=dataset_path, steps_to_run=steps_to_run, random_drop=random_drop
    )
