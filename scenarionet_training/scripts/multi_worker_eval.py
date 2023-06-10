import argparse
import pickle
import json
import os

import numpy as np

from scenarionet_training.scripts.train_nuplan import config
from scenarionet_training.train_utils.callbacks import DrivingCallbacks
from scenarionet_training.train_utils.multi_worker_PPO import MultiWorkerPPO
from scenarionet_training.train_utils.utils import initialize_ray


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--database_path", type=str, required=True)
    parser.add_argument("--id", type=str, default="")
    parser.add_argument("--num_scenarios", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--allowed_more_steps", type=int, default=50)
    parser.add_argument("--max_lateral_dist", type=int, default=2.5)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    file = "eval_{}_{}_{}".format(args.id, os.path.basename(args.ckpt_path), os.path.basename(args.database_path))
    if os.path.exists(file) and not args.overwrite:
        raise FileExistsError("Please remove {} or set --overwrite".format(file))
    initialize_ray(test_mode=True, num_gpus=1)

    config["callbacks"] = DrivingCallbacks
    config["evaluation_num_workers"] = args.num_workers
    config["evaluation_num_episodes"] = args.num_scenarios
    config["metrics_smoothing_episodes"] = args.num_scenarios
    config["custom_eval_function"] = None
    config["num_workers"] = 0
    config["evaluation_config"]["env_config"].update(dict(
        start_scenario_index=args.start_index,
        num_scenarios=args.num_scenarios,
        sequential_seed=True,
        store_map=False,
        store_data=False,
        allowed_more_steps=args.allowed_more_steps,
        # no_map=True,
        max_lateral_dist=args.max_lateral_dist,
        curriculum_level=1,  # disable curriculum
        target_success_rate=1,
        horizon=args.horizon,
        episodes_to_evaluate_curriculum=args.num_scenarios,
        data_directory=args.database_path,
        use_render=False))

    trainer = MultiWorkerPPO(config)
    trainer.restore(args.ckpt_path)

    ret = trainer._evaluate()["evaluation"]
    with open(file + ".json", "w") as f:
        json.dump(ret, f, cls=NumpyEncoder)

    with open(file + ".pkl", "wb+") as f:
        pickle.dump(ret, f)
