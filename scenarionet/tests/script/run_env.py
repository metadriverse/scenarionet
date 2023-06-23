import os

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import get_number_of_scenarios

from scenarionet import SCENARIONET_DATASET_PATH, SCENARIONET_PACKAGE_PATH, TMP_PATH
from scenarionet.builder.utils import merge_database

if __name__ == '__main__':
    dataset_paths = [os.path.join(SCENARIONET_DATASET_PATH, "nuscenes")]
    dataset_paths.append(os.path.join(SCENARIONET_DATASET_PATH, "nuplan"))
    dataset_paths.append(os.path.join(SCENARIONET_DATASET_PATH, "waymo"))
    dataset_paths.append(os.path.join(SCENARIONET_DATASET_PATH, "pg"))

    combine_path = os.path.join(TMP_PATH, "combine")
    merge_database(combine_path, *dataset_paths, exist_ok=True, overwrite=True, try_generate_missing_file=True)

    env = ScenarioEnv(
        {
            "use_render": True,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "show_interface": True,
            "debug": False,
            "show_logo": False,
            "show_fps": False,
            "force_reuse_object_name": True,
            "num_scenarios": get_number_of_scenarios(combine_path),
            "horizon": 1000,
            "no_static_vehicles": True,
            "vehicle_config": dict(
                show_navi_mark=False,
                no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
            "data_directory": combine_path,
        }
    )
    success = []
    env.reset(seed=91)
    while True:
        env.reset(seed=91)
        for t in range(10000):
            o, r, d, _, info = env.step([0, 0])
            assert env.observation_space.contains(o)
            c_lane = env.vehicle.lane
            long, lat, = c_lane.local_coordinates(env.vehicle.position)
            if env.config["use_render"]:
                env.render(text={
                    "scenario index": env.engine.global_seed + env.config["start_scenario_index"],
                })

            if d and info["arrive_dest"]:
                print("scenario index:{}, success".format(env.engine.global_random_seed))
                break
