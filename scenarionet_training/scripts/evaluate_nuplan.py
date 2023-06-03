from scenarionet_training.scripts.train_nuplan import config
from scenarionet_training.train_utils.utils import eval_ckpt

if __name__ == '__main__':
    # 27 29 30 37 39
    ckpt_path = "C:\\Users\\x1\\Desktop\\checkpoint_510\\checkpoint-510"
    scenario_data_path = "D:\\scenarionet_testset\\nuplan_test\\nuplan_test_w_raw"
    num_scenarios = 2000
    start_scenario_index = 0
    horizon = 600
    render = False
    explore = True  # PPO is a stochastic policy, turning off exploration can reduce jitter but may harm performance
    log_interval = 10

    eval_ckpt(config,
              ckpt_path,
              scenario_data_path,
              num_scenarios,
              start_scenario_index,
              horizon,
              render,
              explore,
              log_interval)
