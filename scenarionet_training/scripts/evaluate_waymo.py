from scenarionet_training.scripts.train_waymo import config
from scenarionet_training.train_utils.utils import eval_ckpt

if __name__ == '__main__':
    ckpt_path = "C:\\Users\\x1\\Desktop\\checkpoint_170\\checkpoint-170"
    scenario_data_path = "D:\\scenarionet_testset\\waymo_test_raw_data"
    num_scenarios = 2000
    start_scenario_index = 0
    horizon = 600
    render = True
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
