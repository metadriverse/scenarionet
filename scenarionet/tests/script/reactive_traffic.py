from metadrive.envs.scenario_env import ScenarioEnv

if __name__ == "__main__":
    env = ScenarioEnv(
        {
            "use_render": True,
            # "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "show_interface": False,
            "show_logo": False,
            "show_fps": False,
            "show_mouse": False,
            # "debug": True,
            # "debug_static_world": True,
            # "no_traffic": True,
            # "no_light": True,
            # "debug":True,
            # "no_traffic":True,
            "start_scenario_index": 1,
            # "start_scenario_index": 1000,
            "num_scenarios": 1,
            # "force_reuse_object_name": True,
            # "data_directory": "/home/shady/Downloads/test_processed",
            "horizon": 1000,
            "render_pipeline": True,
            # "reactive_traffic": True,
            "no_static_vehicles": False,
            "force_render_fps": 10,
            "show_policy_mark": True,
            # "show_coordinates": True,
            "vehicle_config": dict(
                show_navi_mark=False,
                no_wheel_friction=False,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
        }
    )
    success = []
    while True:
        env.reset(seed=1)
        env.stop()
        env.main_camera.chase_camera_height = 50
        for i in range(250):
            if i < 50:
                action = [0, -1]
            elif i < 70:
                action = [0.55, 0.5]
            elif i < 80:
                action = [-0.5, 0.5]
            # elif i < 100:
            #     action = [0, -0.4]
            elif i < 110:
                action = [0, -0.1]
            elif i < 130:
                action = [0.3, 0.5]
            else:
                action = [0, -0.5]
            o, r, tm, tc, info = env.step(action)
