import copy
import json
import os.path
from collections import defaultdict

from metadrive.envs.scenario_env import ScenarioEnv

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet_training.train_utils.multi_worker_PPO import MultiWorkerPPO
from scenarionet_training.scripts.train_pg import config
from scenarionet_training.train_utils.utils import initialize_ray, get_time_str

initialize_ray(test_mode=False, num_gpus=1)


def get_eval_config():
    eval_config = copy.deepcopy(config)
    eval_config.pop("evaluation_interval")
    eval_config.pop("evaluation_num_episodes")
    eval_config.pop("evaluation_config")
    eval_config.pop("evaluation_num_workers")
    return eval_config


def get_function(ckpt, explore):
    trainer = MultiWorkerPPO(get_eval_config())
    trainer.restore(ckpt)

    def _f(obs):
        ret = trainer.compute_actions({"default_policy": obs}, explore=explore)
        return ret

    return _f


if __name__ == '__main__':
    # 10/15/20/26/30/31/32
    ckpt_path = "C:\\Users\\x1\\Desktop\\checkpoint_830\\checkpoint-830"
    scenario_data_path = os.path.join(SCENARIONET_DATASET_PATH, "pg_2000")
    num_scenarios = 2000
    start_scenario_index = 0
    horizon = 600
    render = True
    explore = True  # PPO is a stochastic policy, turning off exploration can reduce jitter but may harm performance

    env_config = get_eval_config()["env_config"]
    env_config.update(dict(start_scenario_index=start_scenario_index,
                           num_scenarios=num_scenarios,
                           # sequential_seed=False,
                           curriculum_level=1, # disable curriculum
                           target_success_rate=1,
                           episodes_to_evaluate_curriculum=num_scenarios,
                           data_directory=scenario_data_path,
                           use_render=render))
    env = ScenarioEnv(env_config)

    super_data = defaultdict(list)
    EPISODE_NUM = env.config["num_scenarios"]
    compute_actions = get_function(ckpt_path, explore=explore)

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
        if env.config["use_render"]:
            env.render(text={"reward": r, "seed": env.current_seed})
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
