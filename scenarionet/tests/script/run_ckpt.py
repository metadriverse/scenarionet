import pygame
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.real_data_envs.nuscenes_env import ScenarioEnv
from metadrive.envs.gym_wrapper import createGymWrapper
from scenarionet_training.train_utils.utils import initialize_ray, get_function
from scenarionet_training.scripts.train_nuplan import config

if __name__ == "__main__":
    initialize_ray(test_mode=False, num_gpus=1)
    env = createGymWrapper(ScenarioEnv)(
        {
            # "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
            "data_directory": "D:\\scenarionet_testset\\nuplan_test\\nuplan_test_w_raw",
            "use_render": True,
            # "agent_policy": ReplayEgoCarPolicy,
            "show_interface": False,
            "image_observation": False,
            "show_logo": False,
            "no_traffic": False,
            "no_static_vehicles": False,
            "drivable_region_extension": 15,
            "sequential_seed": True,
            "reactive_traffic": True,
            "show_fps": False,
            "render_pipeline": True,
            "daytime": "07:10",
            "max_lateral_dist": 2,
            "window_size": (1200, 800),
            "camera_dist": 9,
            "start_scenario_index": 5,
            "num_scenarios": 4000,
            "horizon": 1000,
            "store_map": False,
            "vehicle_config": dict(
                show_navi_mark=True,
                # no_wheel_friction=True,
                use_special_color=False,
                image_source="depth_camera",
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=0, distance=50)
            ),
        }
    )

    # env.reset()
    #
    #
    ckpt = "C:\\Users\\x1\\Desktop\\neurips_2023\\exp\\nuplan\\MultiWorkerPPO_ScenarioEnv_2f75c_00003_3_seed=300_2023-06-04_02-14-18\\checkpoint_430\\checkpoint-430"
    policy = get_function(ckpt, True, config)

    def capture():
        env.capture("rgb_deluxe_{}_{}.jpg".format(env.current_seed, t))
        # ret = env.render(
        #     mode="topdown", screen_size=(1600, 900), film_size=(10000, 10000), target_vehicle_heading_up=True
        # )
        # pygame.image.save(ret, "top_down_{}_{}.png".format(env.current_seed, env.episode_step))

    #
    #
    # env.engine.accept("c", capture)

    for i in range(10000):
        o = env.reset()
        for t in range(10000):
            # capture()
            o, r, d, info = env.step(policy(o)["default_policy"])
            if d and info["arrive_dest"]:
                break
