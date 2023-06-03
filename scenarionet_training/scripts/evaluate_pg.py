import os.path

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet_training.scripts.train_pg import config
from scenarionet_training.train_utils.utils import eval_ckpt

if __name__ == '__main__':
    # Merge all evaluate script
    # 10/15/20/26/30/31/32
    ckpt_path = "C:\\Users\\x1\\Desktop\\checkpoint_330\\checkpoint-330"
    scenario_data_path = os.path.join(SCENARIONET_DATASET_PATH, "pg_2000")
    num_scenarios = 2000
    start_scenario_index = 0
    horizon = 600
    render = False
    explore = True  # PPO is a stochastic policy, turning off exploration can reduce jitter but may harm performance
    log_interval = 2

    eval_ckpt(config,
              ckpt_path,
              scenario_data_path,
              num_scenarios,
              start_scenario_index,
              horizon,
              render,
              explore,
              log_interval)
