import pygame

from scenarionet.converter.pg.utils import make_env


def capture():
    env.capture("rgb_deluxe_{}_{}.jpg".format(env.current_seed, t))
    ret = env.render(mode="topdown", screen_size=(1600, 900), film_size=(6000, 6000), target_vehicle_heading_up=True)
    pygame.image.save(ret, "top_down_{}_{}.png".format(env.current_seed, env.episode_step))


if __name__ == "__main__":
    env = make_env(
        0,
        50000,
        extra_config=dict(
            use_render=True,
            show_logo=False,
            show_fps=False,
            show_interface=False,
            drivable_region_extension=15,
            window_size=(1600, 900),
            render_pipeline=True,
            camera_dist=9,
            random_spawn_lane_index=False,
            vehicle_config=dict(show_navi_mark=False),
            daytime="07:10"
        )
    )

    # o = env.reset(seed=0)
    # env.engine.accept("c", capture)
    for s in range(6, 1000):
        env.reset(seed=16)
        for t in range(10000):
            capture()
            o, r, d, info = env.step([0, 0])
            if info["arrive_dest"]:
                break
