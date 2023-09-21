desc = "Load a database to simulator and replay scenarios"

if __name__ == '__main__':
    import logging

    import pkg_resources  # for suppress warning
    import argparse
    import os
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    from metadrive.scenario.utils import get_number_of_scenarios

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--database_path", "-d", required=True, help="The path of the database")
    parser.add_argument("--render", default="none", choices=["none", "2D", "3D", "advanced", "semantic"])
    parser.add_argument("--scenario_index", default=None, type=int, help="Specifying a scenario to run")
    args = parser.parse_args()

    database_path = os.path.abspath(args.database_path)
    num_scenario = get_number_of_scenarios(database_path)
    if args.scenario_index is not None:
        assert args.scenario_index < num_scenario, \
            "The specified scenario index exceeds the scenario range: {}!".format(num_scenario)

    env = ScenarioEnv(
        {
            "use_render": args.render == "3D" or args.render == "advanced",
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "render_pipeline": args.render == "advanced",
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
    for index in range(0, num_scenario if args.scenario_index is not None else 1000000):
        env.reset(seed=index if args.scenario_index is None else args.scenario_index)
        for t in range(10000):
            env.step([0, 0])
            if env.config["use_render"]:
                env.render(
                    text={
                        "scenario index": env.engine.global_seed + env.config["start_scenario_index"],
                        "[": "Load last scenario",
                        "]": "Load next scenario",
                        "r": "Reset current scenario",
                    }
                )

            if args.render == "2D" or args.render == "semantic":
                env.render(
                    film_size=(3000, 3000),
                    semantic_map=args.render == "semantic",
                    target_vehicle_heading_up=False,
                    mode="top_down",
                    text={
                        "scenario index": env.engine.global_seed + env.config["start_scenario_index"],
                    }
                )
            if env.episode_step >= env.engine.data_manager.current_scenario_length:
                print("scenario:{}, success".format(env.engine.global_random_seed))
                break
