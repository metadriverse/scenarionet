import copy
import datetime
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import tqdm
from metadrive.constants import TerminationState
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.envs.gym_wrapper import createGymWrapper
from ray import tune
from ray.tune import CLIReporter

from scenarionet_training.train_utils.multi_worker_PPO import MultiWorkerPPO
from scenarionet_training.wandb_utils import WANDB_KEY_FILE

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def get_api_key_file(wandb_key_file):
    if wandb_key_file is not None:
        default_path = os.path.expanduser(wandb_key_file)
    else:
        default_path = WANDB_KEY_FILE
    if os.path.exists(default_path):
        print("We are using this wandb key file: ", default_path)
        return default_path
    path = os.path.join(root, "scenarionet_training/wandb", "wandb_api_key_file.txt")
    print("We are using this wandb key file: ", path)
    return path


def train(
        trainer,
        config,
        stop,
        exp_name,
        num_seeds=1,
        num_gpus=0,
        test_mode=False,
        suffix="",
        checkpoint_freq=10,
        keep_checkpoints_num=None,
        start_seed=0,
        local_mode=False,
        save_pkl=True,
        custom_callback=None,
        max_failures=0,
        wandb_key_file=None,
        wandb_project=None,
        wandb_team="drivingforce",
        wandb_log_config=True,
        init_kws=None,
        save_dir=None,
        **kwargs
):
    init_kws = init_kws or dict()
    # initialize ray
    if not os.environ.get("redis_password"):
        initialize_ray(test_mode=test_mode, local_mode=local_mode, num_gpus=num_gpus, **init_kws)
    else:
        password = os.environ.get("redis_password")
        assert os.environ.get("ip_head")
        print(
            "We detect redis_password ({}) exists in environment! So "
            "we will start a ray cluster!".format(password)
        )
        if num_gpus:
            print(
                "We are in cluster mode! So GPU specification is disable and"
                " should be done when submitting task to cluster! You are "
                "requiring {} GPU for each machine!".format(num_gpus)
            )
        initialize_ray(address=os.environ["ip_head"], test_mode=test_mode, redis_password=password, **init_kws)

    # prepare config

    if custom_callback:
        callback = custom_callback
    else:
        from scenarionet_training.train_utils.callbacks import DrivingCallbacks
        callback = DrivingCallbacks

    used_config = {
        "seed": tune.grid_search([i * 100 + start_seed for i in range(num_seeds)]) if num_seeds is not None else None,
        "log_level": "DEBUG" if test_mode else "INFO",
        "callbacks": callback
    }
    if custom_callback is False:
        used_config.pop("callbacks")
    if config:
        used_config.update(config)
    config = copy.deepcopy(used_config)

    if isinstance(trainer, str):
        trainer_name = trainer
    elif hasattr(trainer, "_name"):
        trainer_name = trainer._name
    else:
        trainer_name = trainer.__name__

    if not isinstance(stop, dict) and stop is not None:
        assert np.isscalar(stop)
        stop = {"timesteps_total": int(stop)}

    if keep_checkpoints_num is not None and not test_mode:
        assert isinstance(keep_checkpoints_num, int)
        kwargs["keep_checkpoints_num"] = keep_checkpoints_num
        kwargs["checkpoint_score_attr"] = "episode_reward_mean"

    if "verbose" not in kwargs:
        kwargs["verbose"] = 1 if not test_mode else 2

    # This functionality is not supported yet!
    metric_columns = CLIReporter.DEFAULT_COLUMNS.copy()
    progress_reporter = CLIReporter(metric_columns=metric_columns)
    progress_reporter.add_metric_column("success")
    progress_reporter.add_metric_column("coverage")
    progress_reporter.add_metric_column("out")
    progress_reporter.add_metric_column("max_step")
    progress_reporter.add_metric_column("length")
    progress_reporter.add_metric_column("level")
    kwargs["progress_reporter"] = progress_reporter

    if wandb_key_file is not None:
        assert wandb_project is not None
    if wandb_project is not None:
        assert wandb_project is not None
        failed_wandb = False
        try:
            from scenarionet_training.wandb_utils.our_wandb_callbacks import OurWandbLoggerCallback
        except Exception as e:
            # print("Please install wandb: pip install wandb")
            failed_wandb = True

        if failed_wandb:
            from ray.tune.logger import DEFAULT_LOGGERS
            from scenarionet_training.wandb_utils.our_wandb_callbacks_ray100 import OurWandbLogger
            kwargs["loggers"] = DEFAULT_LOGGERS + (OurWandbLogger,)
            config["logger_config"] = {
                "wandb":
                    {
                        "group": exp_name,
                        "exp_name": exp_name,
                        "entity": wandb_team,
                        "project": wandb_project,
                        "api_key_file": get_api_key_file(wandb_key_file),
                        "log_config": wandb_log_config,
                    }
            }
        else:
            kwargs["callbacks"] = [
                OurWandbLoggerCallback(
                    exp_name=exp_name,
                    api_key_file=get_api_key_file(wandb_key_file),
                    project=wandb_project,
                    group=exp_name,
                    log_config=wandb_log_config,
                    entity=wandb_team
                )
            ]

    # start training
    analysis = tune.run(
        trainer,
        name=exp_name,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True if "checkpoint_at_end" not in kwargs else kwargs.pop("checkpoint_at_end"),
        stop=stop,
        config=config,
        max_failures=max_failures if not test_mode else 0,
        reuse_actors=False,
        local_dir=save_dir or ".",
        **kwargs
    )

    # save training progress as insurance
    if save_pkl:
        pkl_path = "{}-{}{}.pkl".format(exp_name, trainer_name, "" if not suffix else "-" + suffix)
        with open(pkl_path, "wb") as f:
            data = analysis.fetch_trial_dataframes()
            pickle.dump(data, f)
            print("Result is saved at: <{}>".format(pkl_path))
    return analysis


import argparse
import logging
import os

import ray


def initialize_ray(local_mode=False, num_gpus=None, test_mode=False, **kwargs):
    os.environ['OMP_NUM_THREADS'] = '1'

    if ray.__version__.split(".")[0] == "1":  # 1.0 version Ray
        if "redis_password" in kwargs:
            redis_password = kwargs.pop("redis_password")
            kwargs["_redis_password"] = redis_password

    ray.init(
        logging_level=logging.ERROR if not test_mode else logging.DEBUG,
        log_to_driver=test_mode,
        local_mode=local_mode,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
        include_dashboard=False,
        **kwargs
    )
    print("Successfully initialize Ray!")
    try:
        print("Available resources: ", ray.available_resources())
    except Exception:
        pass


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-cpus-per-worker", type=float, default=0.5)
    parser.add_argument("--num-gpus-per-trial", type=float, default=0.25)
    parser.add_argument("--test", action="store_true")
    return parser


def setup_logger(debug=False):
    import logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_exp_name(args):
    if args.exp_name != "":
        exp_name = args.exp_name + "_" + get_time_str()
    else:
        exp_name = "TEST"
    return exp_name


def get_eval_config(config):
    eval_config = copy.deepcopy(config)
    eval_config.pop("evaluation_interval")
    eval_config.pop("evaluation_num_episodes")
    eval_config.pop("evaluation_config")
    eval_config.pop("evaluation_num_workers")
    return eval_config


def get_function(ckpt, explore, config):
    trainer = MultiWorkerPPO(get_eval_config(config))
    trainer.restore(ckpt)

    def _f(obs):
        ret = trainer.compute_actions({"default_policy": obs}, explore=explore)
        return ret

    return _f


def eval_ckpt(config,
              ckpt_path,
              scenario_data_path,
              num_scenarios,
              start_scenario_index,
              horizon=600,
              render=False,
              # PPO is a stochastic policy, turning off exploration can reduce jitter but may harm performance
              explore=True,
              log_interval=None,
              ):
    initialize_ray(test_mode=False, num_gpus=1)
    # 27 29 30 37 39
    env_config = get_eval_config(config)["env_config"]
    env_config.update(dict(
        start_scenario_index=start_scenario_index,
        num_scenarios=num_scenarios,
        sequential_seed=True,
        curriculum_level=1,  # disable curriculum
        target_success_rate=1,
        horizon=horizon,
        episodes_to_evaluate_curriculum=num_scenarios,
        data_directory=scenario_data_path,
        use_render=render))
    env = createGymWrapper(ScenarioEnv)(env_config)

    super_data = defaultdict(list)
    EPISODE_NUM = env.config["num_scenarios"]
    compute_actions = get_function(ckpt_path, explore=explore, config=config)

    o = env.reset()
    assert env.current_seed == start_scenario_index, "Wrong start seed!"

    total_cost = 0
    total_reward = 0
    success_rate = 0
    ep_cost = 0
    ep_reward = 0
    success_flag = False
    step = 0

    def log_msg():
        print("CKPT:{} | success_rate:{}, mean_episode_reward:{}, mean_episode_cost:{}".format(epi_num,
                                                                                               success_rate / epi_num,
                                                                                               total_reward / epi_num,
                                                                                               total_cost / epi_num))

    for epi_num in tqdm.tqdm(range(0, EPISODE_NUM)):
        step += 1
        action_to_send = compute_actions(o)["default_policy"]
        o, r, d, info = env.step(action_to_send)
        if env.config["use_render"]:
            env.render(text={"reward": r})
        total_reward += r
        ep_reward += r
        total_cost += info["cost"]
        ep_cost += info["cost"]
        if d or step > horizon:
            if info["arrive_dest"]:
                success_rate += 1
                success_flag = True
            o = env.reset()

            super_data[0].append(
                {"reward": ep_reward,
                 "success": success_flag,
                 "out_of_road": info[TerminationState.OUT_OF_ROAD],
                 "cost": ep_cost,
                 "seed": env.current_seed,
                 "route_completion": info["route_completion"]})

            ep_cost = 0.0
            ep_reward = 0.0
            success_flag = False
            step = 0

            if log_interval is not None and epi_num % log_interval == 0:
                log_msg()
    if log_interval is not None:
        log_msg()
    del compute_actions
    env.close()
    with open("eval_ret_{}_{}_{}.json".format(start_scenario_index,
                                              start_scenario_index + num_scenarios,
                                              get_time_str()), "w") as f:
        json.dump(super_data, f)
    return super_data
