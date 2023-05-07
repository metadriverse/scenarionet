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


def verify_loading_into_metadrive(dataset_path, result_save_dir, steps_to_run=300, num_workers=8):
    if result_save_dir is not None:
        assert os.path.exists(result_save_dir) and os.path.isdir(
            result_save_dir), "Argument result_save_dir must be an existing dir"
    num_scenario = get_number_of_scenarios(dataset_path)
    argument_list = []

    func = partial(_loading_into_metadrive,
                   dataset_path=dataset_path,
                   result_save_dir=result_save_dir,
                   steps_to_run=steps_to_run)

    num_scenario_each_worker = int(num_scenario / num_workers)
    for i in range(num_workers):
        if i == num_workers - 1:
            num_scenario_each_worker = num_scenario - num_scenario_each_worker * (num_workers - 1)
        argument_list.append([i * num_scenario_each_worker, num_scenario_each_worker])

    with multiprocessing.Pool(num_workers) as p:
        result = list(p.imap(func, argument_list))
    if all([i[0] for i in result]):
        print("All scenarios can be loaded successfully!")
    else:
        print("Fail to load all scenarios, see log for more details! Number of logs: {}".format(num_workers))


def _loading_into_metadrive(start_scenario_index, num_scenario, dataset_path, result_save_dir, steps_to_run):
    print("================ Begin Scenario Loading Verification for {} ================ \n".format(result_save_dir))
    assert os.path.exists(result_save_dir) and os.path.isdir(
        result_save_dir), "Argument result_save_dir must be an existing dir"
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
    logging.disable(logging.WARNING)
    error_files = []
    try:
        for i in tqdm.tqdm(range(num_scenario),
                           desc="Scenarios: {}-{}".format(start_scenario_index, start_scenario_index + num_scenario)):
            env.reset(force_seed=i)
            for i in range(steps_to_run):
                o, r, d, i = env.step([0, 0])
                if d:
                    assert i["arrive_dest"], "Can not arrive destination"
    except Exception as e:
        file_name = env.engine.data_manager.summary_lookup[i]
        file_path = os.path.join(dataset_path, env.engine.data_manager.mapping[file_name], file_name)
        error_file = {"seed": i, "file_path": file_path, "error": e}
        error_files.append(error_file)
        logger.warning("\n Scenario Error, seed: {}, file_path: {}.\n Error message: {}".format(i, file_path, e))
        success = False
    finally:
        env.close()
    if result_save_dir is not None:
        file_name = "error_scenarios_{}_{}_{}.json".format(os.path.basename(dataset_path),
                                                           start_scenario_index,
                                                           start_scenario_index + num_scenario)
        with open(os.path.join(result_save_dir, file_name), "w+") as f:
            json.dump(error_files, f)
    return success, error_files
