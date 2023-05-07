import json
import logging
import multiprocessing
import os

logger = logging.getLogger(__name__)
import tqdm
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import get_number_of_scenarios
from functools import partial


def verify_loading_into_metadrive(dataset_path, result_save_dir, steps_to_run=1000, num_workers=8):
    if result_save_dir is not None:
        assert os.path.exists(result_save_dir) and os.path.isdir(
            result_save_dir), "Argument result_save_dir must be an existing dir"
    num_scenario = get_number_of_scenarios(dataset_path)
    argument_list = []

    func = partial(loading_wrapper,
                   dataset_path=dataset_path,
                   steps_to_run=steps_to_run)

    num_scenario_each_worker = int(num_scenario // num_workers)
    for i in range(num_workers):
        if i == num_workers - 1:
            scenario_num = num_scenario - num_scenario_each_worker * (num_workers - 1)
        else:
            scenario_num = num_scenario_each_worker
        argument_list.append([i * num_scenario_each_worker, scenario_num])

    with multiprocessing.Pool(num_workers) as p:
        all_result = list(p.imap(func, argument_list))
    result = all([i[0] for i in all_result])
    logs = []
    for _, log in all_result:
        logs += log

    if result_save_dir is not None:
        file_name = "error_scenarios_for_{}.json".format(os.path.basename(dataset_path))
        with open(os.path.join(result_save_dir, file_name), "w+") as f:
            json.dump(logs, f, indent=4)

    if result:
        print("All scenarios can be loaded successfully!")
    else:
        print("Fail to load all scenarios, see log for more details! Number of failed scenarios: {}".format(len(logs)))
    return result, logs


def loading_into_metadrive(start_scenario_index, num_scenario, dataset_path, steps_to_run):
    print("================ Begin Scenario Loading Verification for scenario {}-{} ================ \n".format(
        start_scenario_index, num_scenario + start_scenario_index))
    success = True
    env = ScenarioEnv(
        {
            "agent_policy": ReplayEgoCarPolicy,
            "num_scenarios": num_scenario,
            "horizon": 1000,
            "start_scenario_index": start_scenario_index,
            "no_static_vehicles": False,
            "data_directory": dataset_path,
        }
    )
    logging.disable(logging.INFO)
    error_files = []
    try:
        for scenario_index in tqdm.tqdm(range(start_scenario_index, start_scenario_index + num_scenario),
                                        desc="Scenarios: {}-{}".format(start_scenario_index,
                                                                       start_scenario_index + num_scenario)):
            env.reset(force_seed=scenario_index)
            arrive = False
            for _ in range(steps_to_run):
                o, r, d, info = env.step([0, 0])
                if d and info["arrive_dest"]:
                    arrive = True
            assert arrive, "Can not arrive destination"
    except Exception as e:
        file_name = env.engine.data_manager.summary_lookup[scenario_index]
        file_path = os.path.join(dataset_path, env.engine.data_manager.mapping[file_name], file_name)
        error_file = {"scenario_index": scenario_index, "file_path": file_path, "error": str(e)}
        error_files.append(error_file)
        logger.warning("\n Scenario Error, "
                       "scenario_index: {}, file_path: {}.\n Error message: {}".format(scenario_index, file_path,
                                                                                       str(e)))
        success = False
    finally:
        env.close()
    return success, error_files


def loading_wrapper(arglist, dataset_path, steps_to_run):
    assert len(arglist) == 2, "Too much arguments!"
    return loading_into_metadrive(arglist[0],
                                  arglist[1],
                                  dataset_path=dataset_path,
                                  steps_to_run=steps_to_run)
