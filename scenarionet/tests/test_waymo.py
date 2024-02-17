import urllib.request
import shutil
import argparse
import logging
import os

from scenarionet import SCENARIONET_DATASET_PATH, SCENARIONET_REPO_PATH
from scenarionet.converter.utils import write_to_directory
from scenarionet.converter.waymo.utils import convert_waymo_scenario, get_waymo_scenarios, preprocess_waymo_scenarios
import logging

import pkg_resources  # for suppress warning
import argparse
import os
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import get_number_of_scenarios


def test_waymo_and_sim():
    url = "https://github.com/metadriverse/scenarionet/releases/download/releases%2F0.01/waymo_test_data"
    waymo_data_directory = os.path.join(SCENARIONET_DATASET_PATH, "waymo_raw")
    if os.path.exists(waymo_data_directory):
        shutil.rmtree(waymo_data_directory)
    os.makedirs(waymo_data_directory)
    urllib.request.urlretrieve(url, os.path.join(waymo_data_directory, "training_20s.tfrecord-00000-of-01000"))
    #
    dataset_name = "waymo"
    output_path = "waymo_test_data"
    version = "v1.2"
    #
    files = get_waymo_scenarios(waymo_data_directory, 0, 1)

    write_to_directory(
        convert_func=convert_waymo_scenario,
        scenarios=files,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        overwrite=True,
        num_workers=4,
        preprocess=preprocess_waymo_scenarios,
    )
    database_path = os.path.abspath(output_path)
    num_scenario = get_number_of_scenarios(database_path)

    env = ScenarioEnv(
        {
            "use_render": False,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "render_pipeline": False,
            "show_interface": True,
            # "reactive_traffic": args.reactive,
            "show_logo": False,
            "show_fps": False,
            "log_level": logging.CRITICAL,
            "num_scenarios": num_scenario,
            "interface_panel": [],
            "horizon": 1000,
            "vehicle_config": dict(
                show_navi_mark=True,
                show_line_to_dest=False,
                show_dest_mark=False,
                no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
            "data_directory": database_path,
        }
    )
    for index in range(0, 20):
        print(index)
        env.reset(seed=index)
        for t in range(10000):
            env.step([0, 0])
            env.render(
                film_size=(3000, 3000),
                semantic_map=True,
                target_vehicle_heading_up=False,
                window=False,
                mode="top_down",
                text={
                    "scenario index": env.engine.global_seed + env.config["start_scenario_index"],
                }
            )
            if env.episode_step >= env.engine.data_manager.current_scenario_length:
                print("scenario:{}, success".format(env.engine.global_random_seed))
                break


if __name__ == '__main__':
    test_waymo_and_sim()
