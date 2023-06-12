import pygame

from scenarionet.converter.pg.utils import make_env


def capture():
    env.capture()
    ret = env.render(mode="topdown", screen_size=(1600, 900), film_size=(2000, 2000), track_target_vehicle=True)
    pygame.image.save(ret, "top_down_{}_step_{}.png".format(env.current_seed, env.episode_step))


if __name__ == "__main__":
    env = make_env(0, 50000, extra_config=dict(use_render=True,
                                               show_logo=False,
                                               show_fps=False,
                                               show_interface=False,
                                               drivable_region_extension=15,
                                               window_size=(1600, 900),
                                               render_pipeline=True,
                                               camera_dist=9,
                                               vehicle_config=dict(show_navi_mark=False),
                                               daytime="07:10"
                                               ))

    o = env.reset(force_seed=0)
    env.engine.accept("c", capture)
    for s in range(100000):
        env.reset(force_seed=s)
        while True:
            o, r, d, info = env.step([0, 0])
            if d:
                break
