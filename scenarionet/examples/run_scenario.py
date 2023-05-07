import os

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

from scenarionet import SCENARIONET_DATASET_PATH

if __name__ == '__main__':
    dataset_path = os.path.join(SCENARIONET_DATASET_PATH, "nuscenes_no_mapping_test")

    env = ScenarioEnv(
        {
            "use_render": True,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "show_interface": True,
            "show_logo": False,
            "show_fps": False,
            "num_scenarios": 10,
            "horizon": 1000,
            "no_static_vehicles": True,
            "vehicle_config": dict(
                show_navi_mark=False,
                no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
            "data_directory": dataset_path,
        }
    )
    success = []
    env.reset(force_seed=0)
    while True:
        env.reset(force_seed=env.current_seed + 1)
        for t in range(10000):
            o, r, d, info = env.step([0, 0])
            assert env.observation_space.contains(o)
            c_lane = env.vehicle.lane
            long, lat, = c_lane.local_coordinates(env.vehicle.position)
            # if env.config["use_render"]:
            env.render(
                text={
                    "seed": env.engine.global_seed + env.config["start_scenario_index"],
                }
            )

            if d and info["arrive_dest"]:
                print("seed:{}, success".format(env.engine.global_random_seed))
                break
