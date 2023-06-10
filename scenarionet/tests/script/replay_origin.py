import time

import pygame
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

NuScenesEnv = ScenarioEnv

if __name__ == "__main__":
    env = NuScenesEnv(
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
            # "debug": True,
            "render_pipeline": True,
            "daytime": "19:30",
            "window_size": (1600, 900),
            "camera_dist": 9,
            # "camera_height": 1.5,
            # "camera_pitch": None,
            # "camera_fov": 60,
            "start_scenario_index": 0,
            "num_scenarios": 4,
            # "force_reuse_object_name": True,
            # "data_directory": "/home/shady/Downloads/test_processed",
            "horizon": 1000,
            # "no_static_vehicles": True,
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
                # rgb_camera=(1600, 900),
                # depth_camera=(1600, 900, True),
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=12, distance=50)
            ),
            "data_directory": "D:\\code\\scenarionet\\scenarionet\\tests\\script\\waymo_scenes_adv"
        }
    )

    # 0,1,3,4,5,6

    success = []
    reset_num = 0
    start = time.time()
    reset_used_time = 0
    s = 0

    env.reset()


    def capture():
        env.capture()
        ret = env.render(mode="topdown", screen_size=(1600, 900), film_size=(7000, 7000), track_target_vehicle=True)
        pygame.image.save(ret, "top_down_{}.png".format(env.current_seed))


    env.engine.accept("c", capture)

    while True:
        # for i in range(10):
        start_reset = time.time()
        env.reset()

        reset_used_time += time.time() - start_reset
        reset_num += 1
        for t in range(10000):
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
