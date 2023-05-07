import os

import tqdm
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import get_number_of_scenarios


def verify_loading_into_metadrive(dataset_path):
    scenario_num = get_number_of_scenarios(dataset_path)

    env = ScenarioEnv(
        {
            "agent_policy": ReplayEgoCarPolicy,
            "num_scenarios": scenario_num,
            "horizon": 1000,
            "no_static_vehicles": False,
            "data_directory": dataset_path,
        }
    )
    try:
        for i in tqdm.tqdm(range(scenario_num)):
            env.reset(force_seed=i)
    except Exception as e:
        file_name = env.engine.data_manager.summary_lookup[i]
        file_path = os.path.join(dataset_path, env.engine.data_manager.mapping[file_name], file_name)
        raise ValueError("Scenario Error, seed: {}, file_path: {}. "
                         "\n Error message: {}".format(i, file_path, e))
