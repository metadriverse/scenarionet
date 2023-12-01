import pygame
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

NuScenesEnv = ScenarioEnv

if __name__ == "__main__":
    env = NuScenesEnv(
        {
            "use_render": True,
            "agent_policy": ReplayEgoCarPolicy,
            "show_interface": True,
            "interface_panel": ["semantic_camera"],
            # "need_lane_localization": False,
            "show_logo": False,
            "no_traffic": False,
            "drivable_region_extension": 15,
            "sequential_seed": True,
            "reactive_traffic": False,
            "show_fps": False,
            # "debug": True,
            "render_pipeline": False,
            "daytime": "08:10",
            "window_size": (800, 450),
            "camera_dist": 0.8,
            "camera_height": 1.5,
            "camera_pitch": None,
            "camera_fov": 60,
            "start_scenario_index": 0,
            "num_scenarios": 10,
            # "force_reuse_object_name": True,
            # "data_directory": "/home/shady/Downloads/test_processed",
            "horizon": 1000,
            "no_static_vehicles": False,
            # "show_policy_mark": True,
            # "show_coordinates": True,
            # "force_destroy": True,
            # "default_vehicle_in_traffic": True,
            "default_vehicle_in_traffic": True,
            "sensors": dict(semantic_camera=(SemanticCamera, 1600, 900)),
            "vehicle_config": dict(
                show_navi_mark=False,
                use_special_color=False,
                image_source="semantic_camera",
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=12, distance=50)
            ),
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),

            # "image_observation": True,
        }
    )

    # 0,1,3,4,5,6
    # for seed in range(10):
    while True:
        env.reset(seed=6)
        for t in range(10000):
            # env.capture("rgb_deluxe_{}_{}.jpg".format(env.current_seed, t))
            # ret = env.render(
            #     mode="topdown", screen_size=(1600, 900), film_size=(9000, 9000), target_vehicle_heading_up=True
            # )
            # pygame.image.save(ret, "top_down_{}_{}.png".format(env.current_seed, t))
            # env.vehicle.get_camera("depth_camera").save_image(env.vehicle, "depth_{}.jpg".format(t))
            # env.vehicle.get_camera("rgb_camera").save_image(env.vehicle, "rgb_{}.jpg".format(t))
            if t == 100:
                # env.engine.get_sensor("semantic_camera").save_image(env.vehicle, "depth_{}.jpg".format(t))
                env.engine.get_sensor("semantic_camera").save_image(env.vehicle, "semantic_{}.jpg".format(t))
                break
            env.step([1, 0.88])
            # if d:
