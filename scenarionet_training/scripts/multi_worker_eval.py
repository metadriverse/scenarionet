import argparse
import json
import multiprocessing
import os
from functools import partial

from scenarionet_training.scripts.train_nuplan import config
from scenarionet_training.train_utils.utils import eval_ckpt
from scenarionet_training.train_utils.utils import get_time_str


def _eval_single_wrapper(start_index, known_kwargs):
    return eval_ckpt(start_scenario_index=start_index, **known_kwargs)


if __name__ == '__main__':
    # 27 29 30 37 39
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--database_path", type=str, required=True)
    parser.add_argument("--num_scenarios", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--explore", type=bool, default=True)
    parser.add_argument("--log_interval", type=int, default=None)

    args = parser.parse_args()
    os.makedirs("eval_ret_{}".format(os.path.basename(args.ckpt_path)), exist_ok=False)

    num_files = args.num_scenarios
    num_workers = args.num_workers
    num_files_each_worker = int(num_files / num_workers)
    assert num_files_each_worker * num_workers == num_files, "num_scenarios should be dividable by num_workers!"
    argument_list = []
    for i in range(num_workers):
        argument_list.append(args.start_index + i * num_files_each_worker)

    _eval_func_wrapper = partial(_eval_single_wrapper,
                                 known_kwagrs=dict(config=config,
                                                   ckpt_path=args.ckpt_path,
                                                   scenario_data_path=args.database_path,
                                                   num_scenarios=args.num_scenarios,
                                                   # start_scenario_index=start_index,
                                                   horizon=args.horizon,
                                                   render=args.render,
                                                   explore=args.explore,
                                                   log_interval=args.log_interval)

                                 )

    # Run, workers and process result from worker
    with multiprocessing.Pool(num_workers, maxtasksperchild=10) as p:
        all_ret = list(p.imap(_eval_func_wrapper, argument_list))
        # call ret to block the process

    final_result = {}
    for ret in all_ret:
        final_result.update(ret)

    with open("merge_eval_ret_{}_{}_{}.json".format(args.start_index,
                                                    args.start_index + args.num_scenarios,
                                                    get_time_str()), "w") as f:
        json.dump(final_result, f)
