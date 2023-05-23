import json
import os.path
from collections import defaultdict

from metadrive.envs.scenario_env import ScenarioEnv

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet_training.train.multi_worker_PPO import MultiWorkerPPO
from scenarionet_training.train.utils import initialize_ray, get_time_str

initialize_ray(test_mode=False, num_gpus=1)


def get_function(ckpt):
    trainer = MultiWorkerPPO(config=dict(
        env=ScenarioEnv,
        env_config=dict(
            # scenario
            start_scenario_index=0,
            num_scenarios=8000,
            data_directory=os.path.join(SCENARIONET_DATASET_PATH, "pg_2000"),
            sequential_seed=True,

            # traffic & light
            reactive_traffic=False,
            no_static_vehicles=True,
            no_light=True,

            # curriculum training
            curriculum_level=40,
            target_success_rate=0.85,

            # training
            horizon=None,
            use_lateral_reward=True,
        ),

        # ===== Training =====
        model=dict(fcnet_hiddens=[512, 256, 128]),
        horizon=600,
        num_sgd_iter=20,
        lr=5e-5,
        rollout_fragment_length=500,
        sgd_minibatch_size=100,
        train_batch_size=40000,
        num_gpus=0.5,
        num_cpus_per_worker=0.4,
        num_cpus_for_driver=1,
        num_workers=10,
        framework="tf"
    ))

    trainer.restore(ckpt)

    def _f(obs):
        ret = trainer.compute_actions({"default_policy": obs}, explore=True)
        return ret

    return _f


if __name__ == '__main__':
    ckpt_path = "C:\\Users\\x1\\Desktop\\checkpoint_210\\checkpoint-210"
    scenario_data_path = os.path.join(SCENARIONET_DATASET_PATH, "pg_2000")
    num_scenarios = 2000
    start_scenario_index = 0
    horizon = 600

    env = ScenarioEnv(dict(
        use_render=False,
        # max_lateral_dist=5,
        # scenario
        start_scenario_index=start_scenario_index,
        num_scenarios=num_scenarios,
        data_directory=scenario_data_path,
        sequential_seed=True,

        # traffic & light
        reactive_traffic=False,
        no_static_vehicles=True,
        no_light=True,

        # curriculum training
        curriculum_level=1,
        target_success_rate=1,

        # training
        horizon=None,
        use_lateral_reward=True,
    ), )

    super_data = defaultdict(list)
    EPISODE_NUM = env.config["num_scenarios"]
    compute_actions = get_function(ckpt_path)

    o = env.reset()
    epi_num = 0

    total_cost = 0
    total_reward = 0
    success_rate = 0
    ep_cost = 0
    ep_reward = 0
    success_flag = False
    step = 0


    def log_msg():
        print("CKPT:{} | success_rate:{}, mean_episode_reward:{}, mean_episode_cost:{}".format(0,
                                                                                               success_rate / epi_num,
                                                                                               total_reward / epi_num,
                                                                                               total_cost / epi_num))


    while True:
        step += 1
        action_to_send = compute_actions(o)["default_policy"]
        o, r, d, info = env.step(action_to_send)
        total_reward += r
        ep_reward += r
        total_cost += info["cost"]
        ep_cost += info["cost"]
        if d or step > horizon:
            if info["arrive_dest"]:
                success_rate += 1
                success_flag = True
            epi_num += 1
            if epi_num > EPISODE_NUM:
                break
            else:
                o = env.reset()

            super_data[0].append(
                {"reward": ep_reward, "success": success_flag, "cost": ep_cost, "seed": env.current_seed})

            ep_cost = 0.0
            ep_reward = 0.0
            success_flag = False
            step = 0

            if epi_num % 100 == 0:
                log_msg()

    log_msg()
    del compute_actions
    env.close()
    try:
        with open("eval_ret_{}.json".format(get_time_str()), "w") as f:
            json.dump(super_data, f)
    except:
        pass
    print(super_data)
