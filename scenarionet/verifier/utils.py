import json
import logging
import os

logger = logging.getLogger(__name__)
import tqdm
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import get_number_of_scenarios


def verify_loading_into_metadrive(dataset_path, result_save_dir=None, steps_to_run=0):
    print("================ Begin Scenario Loading Verification for {} ================ \n".format(result_save_dir))
    scenario_num = get_number_of_scenarios(dataset_path)
    if result_save_dir is not None:
        assert os.path.exists(result_save_dir) and os.path.isdir(
            result_save_dir), "Argument result_save_dir must be an existing dir"
    success = True
    env = ScenarioEnv(
        {
            "agent_policy": ReplayEgoCarPolicy,
            "num_scenarios": scenario_num,
            "horizon": 1000,
            "no_static_vehicles": False,
            "data_directory": dataset_path,
        }
    )
    logging.disable(logging.WARNING)
    error_files = []
    try:
        for i in tqdm.tqdm(range(scenario_num)):
            env.reset(force_seed=i)
            for i in range(steps_to_run):
                env.step([0, 0])
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
        with open(os.path.join(result_save_dir, "error_scenarios_{}.json".format(os.path.basename(dataset_path))),
                  "w+") as f:
            json.dump(error_files, f)
    return success, error_files
