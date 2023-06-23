import pygame
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

if __name__ == "__main__":
    env = ScenarioEnv(
        {
            "use_render": True,
            "agent_policy": ReplayEgoCarPolicy,
            "show_interface": False,
            "image_observation": False,
            "show_logo": False,
            "no_traffic": False,
            "drivable_region_extension": 15,
            "sequential_seed": True,
            "reactive_traffic": False,
            "show_fps": False,
            "render_pipeline": True,
            "daytime": "07:10",
            "window_size": (1600, 900),
            "camera_dist": 9,
            "start_scenario_index": 1000,
            "num_scenarios": 4000,
            "horizon": 1000,
            "store_map": False,
            "vehicle_config": dict(
                show_navi_mark=False,
                no_wheel_friction=True,
                use_special_color=False,
                image_source="depth_camera",
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=12, distance=50)
            ),
            "data_directory": "D:\\scenarionet_testset\\nuplan_test\\nuplan_test_w_raw"
        }
    )

    # env.reset()
    #
    #
    def capture():
        env.capture("rgb_deluxe_{}_{}.jpg".format(env.current_seed, t))
        ret = env.render(
            mode="topdown", screen_size=(1600, 900), film_size=(10000, 10000), target_vehicle_heading_up=True
        )
        pygame.image.save(ret, "top_down_{}_{}.png".format(env.current_seed, env.episode_step))

    #
    #
    # env.engine.accept("c", capture)

    # for seed in [1001, 1002, 1005, 1011]:
    env.reset(seed=1020)
    for t in range(10000):
        capture()
        env.step([1, 0.88])
        if env.episode_step >= env.engine.data_manager.current_scenario_length:
            break
