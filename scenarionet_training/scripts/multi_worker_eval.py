import argparse
import json
import os

from scenarionet_training.scripts.train_nuplan import config
from scenarionet_training.train_utils.multi_worker_PPO import MultiWorkerPPO
from scenarionet_training.train_utils.utils import initialize_ray

if __name__ == '__main__':
    # 27 29 30 37 39
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--database_path", type=str, required=True)
    parser.add_argument("--num_scenarios", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    file = "eval_ret_{}.json".format(os.path.basename(args.ckpt_path))
    if os.path.exists(file) and not args.overwrite:
        raise FileExistsError("Please remove {} or set --overwrite".format(file))
    initialize_ray(test_mode=False, num_gpus=1)

    config["evaluation_config"]["evaluation_num_workers"] = args.num_workers
    config["evaluation_config"]["evaluation_num_episodes"] = args.num_scenarios
    config["evaluation_config"]["metrics_smoothing_episodes"] = args.num_scenarios
    config["num_workers"] = 0
    config["evaluation_config"]["env_config"].update(dict(
        start_scenario_index=args.start_index,
        num_scenarios=args.num_scenarios,
        sequential_seed=True,
        curriculum_level=1,  # disable curriculum
        target_success_rate=1,
        horizon=args.horizon,
        episodes_to_evaluate_curriculum=args.num_scenarios,
        data_directory=args.database_path,
        use_render=False))

    trainer = MultiWorkerPPO(config)
    trainer.restore(args.ckpt_path)
    ret = trainer._evaluate()["evaluation"]
    with open(file, "w") as file:
        json.dump(ret, file)
