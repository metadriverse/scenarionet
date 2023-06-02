import time
import pygame
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

NuScenesEnv = ScenarioEnv

if __name__ == "__main__":
    env = NuScenesEnv(
        {
            "use_render": True,
            # "no_map": False,
            "agent_policy": ReplayEgoCarPolicy,
            # "manual_control": True,
            "show_interface": False,
            # "need_lane_localization": False,
            "image_observation": True,
            "show_logo": False,
            "no_traffic": False,
            "sequential_seed": True,
            # "debug_static_world": True,
            # "sequential_seed": True,
            "reactive_traffic": True,
            # "curriculum_level": 1,
            "show_fps": False,
            # "debug": True,
            # "no_static_vehicles": True,
            # "pstats": True,
            # "render_pipeline": True,
            "daytime": "11:01",
            # "no_traffic": True,
            # "no_light": False,
            # "debug":True,
            # Make video
            # "episodes_to_evaluate_curriculum": 5,
            "window_size": (1600, 900),
            "camera_dist": 0.8,
            "camera_height": 1.5,
            "camera_pitch": None,
            "camera_fov": 60,
            "force_render_fps": 10,
            "start_scenario_index": 0,
            "num_scenarios": 10,
            # "force_reuse_object_name": True,
            # "data_directory": "/home/shady/Downloads/test_processed",
            "horizon": 1000,
            "no_static_vehicles": True,
            # "show_policy_mark": True,
            # "show_coordinates": True,
            # "force_destroy": True,
            # "default_vehicle_in_traffic": True,
            "vehicle_config": dict(
                # light=True,
                # random_color=True,
                show_navi_mark=False,
                use_special_color=False,
                image_source="depth_camera",
                rgb_camera=(1600, 900),
                depth_camera=(1600, 900, True),
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=12, distance=50)
            ),
            "data_directory": AssetLoader.file_path("nuscenes", return_raw_style=False),
        }
    )

    # 0,1,3,4,5,6

    success = []
    reset_num = 0
    start = time.time()
    reset_used_time = 0
    s = 0
    while True:
        # for i in range(10):
        start_reset = time.time()
        env.reset(force_seed=0)

        reset_used_time += time.time() - start_reset
        reset_num += 1
        for t in range(10000):
            if t==30:
                # env.capture("camera_deluxe.jpg")
                # ret = env.render(mode="topdown", screen_size=(1600, 900), film_size=(5000, 5000), track_target_vehicle=True)
                # pygame.image.save(ret, "top_down.png")
                env.vehicle.get_camera("depth_camera").save_image(env.vehicle, "camera.jpg")
            o, r, d, info = env.step([1, 0.88])
            assert env.observation_space.contains(o)
            s += 1
            # if env.config["use_render"]:
            #     env.render(text={"seed": env.current_seed,
            #                      # "num_map": info["num_stored_maps"],
            #                      "data_coverage": info["data_coverage"],
            #                      "reward": r,
            #                      "heading_r": info["step_reward_heading"],
            #                      "lateral_r": info["step_reward_lateral"],
            #                      "smooth_action_r": info["step_reward_action_smooth"]})
            if d:
                print(
                    "Time elapse: {:.4f}. Average FPS: {:.4f}, AVG_Reset_time: {:.4f}".format(
                        time.time() - start, s / (time.time() - start - reset_used_time),
                        reset_used_time / reset_num
                    )
                )
                print("seed:{}, success".format(env.engine.global_random_seed))
                print(list(env.engine.curriculum_manager.recent_success.dict.values()))
                break
