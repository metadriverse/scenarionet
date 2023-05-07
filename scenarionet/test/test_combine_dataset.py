import os
import os.path

import tqdm
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

from scenarionet import SCENARIONET_PACKAGE_PATH
from scenarionet.builder.utils import combine_multiple_dataset, read_dataset_summary, read_scenario


def test_combine_multiple_dataset():
    dataset_name = "nuscenes"
    original_dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "test", "test_dataset", dataset_name)
    dataset_paths = [original_dataset_path + "_{}".format(i) for i in range(5)]

    output_path = os.path.join(SCENARIONET_PACKAGE_PATH, "test", "combine")
    combine_multiple_dataset(output_path,
                             *dataset_paths,
                             force_overwrite=True,
                             try_generate_missing_file=True)
    dataset_paths.append(output_path)
    for dataset_path in dataset_paths:
        summary, sorted_scenarios, mapping = read_dataset_summary(dataset_path)
        for scenario_file in sorted_scenarios:
            read_scenario(os.path.join(dataset_path, mapping[scenario_file], scenario_file))

    env = ScenarioEnv({"agent_policy": ReplayEgoCarPolicy,
                       "num_scenarios": 10,
                       "horizon": 1000,
                       "data_directory": output_path})
    try:
        for i in tqdm.tqdm(range(10), desc="Test env loading"):
            env.reset(force_seed=i)
    finally:
        env.close()


if __name__ == '__main__':
    test_combine_multiple_dataset()
