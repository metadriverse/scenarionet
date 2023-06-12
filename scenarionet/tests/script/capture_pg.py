import pygame
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger

if __name__ == "__main__":
    setup_logger(True)
    env = MetaDriveEnv(
        {
            "num_scenarios": 10,
            "traffic_density": 0.15,
            "traffic_mode": "hybrid",
            "start_seed": 74,
            "show_interface": False,
            "cull_scene": False,
            "random_spawn_lane_index": False,
            "random_lane_width": False,
            "random_agent_model": False,
            "manual_control": True,
            "use_render": True,
            "accident_prob": 0.5,
            "decision_repeat": 5,
            "interface_panel": [],
            "need_inverse_traffic": False,
            "rgb_clip": True,
            "map": 2,
            "random_traffic": False,
            "driving_reward": 1.0,
            "force_destroy": False,
            "show_fps": False,
            "render_pipeline": True,
            "window_size": (1600, 900),
            "camera_dist": 9,
            # "camera_pitch": 30,
            # "camera_height": 1,
            # "camera_smooth": False,
            # "camera_height": -1,
            "vehicle_config": {
                "enable_reverse": False,
                "spawn_velocity_car_frame": True,
                "spawn_lane_index": None,
                "show_navi_mark": False,
            },
        }
    )

    o = env.reset()

    def capture():
        env.capture()
        ret = env.render(mode="topdown", screen_size=(1600, 900), film_size=(2000, 2000), track_target_vehicle=True)
        pygame.image.save(ret, "top_down_{}.png".format(env.current_seed))

    env.engine.accept("c", capture)
    for s in range(1, 100000):
        o, r, d, info = env.step([0, 0])

