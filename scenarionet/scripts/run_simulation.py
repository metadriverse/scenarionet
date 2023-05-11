import pkg_resources # for suppress warning
import argparse
import os
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import get_number_of_scenarios

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-d", required=True, help="The path of the dataset")
    parser.add_argument("--render", action="store_true", help="Enable 3D rendering")
    parser.add_argument("--scenario_index", default=None, type=int, help="Specifying a scenario to run")
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset_path)
    num_scenario = get_number_of_scenarios(dataset_path)
    if args.scenario_index is not None:
        assert args.scenario_index < num_scenario, \
            "The specified scenario index exceeds the scenario range: {}!".format(num_scenario)

    env = ScenarioEnv(
        {
            "use_render": args.render,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "show_interface": True,
            "show_logo": False,
            "show_fps": False,
            "num_scenarios": num_scenario,
            "horizon": 1000,
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
    for seed in range(num_scenario if args.scenario_index is not None else 1000000):
        env.reset(force_seed=seed if args.scenario_index is None else args.scenario_index)
        for t in range(10000):
            o, r, d, info = env.step([0, 0])
            if env.config["use_render"]:
                env.render(text={
                    "seed": env.engine.global_seed + env.config["start_scenario_index"],
                })

            if d and info["arrive_dest"]:
                print("scenario:{}, success".format(env.engine.global_random_seed))
                break
