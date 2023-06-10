"""
Procedure to use wandb:

1. Logup in wandb: https://wandb.ai/
2. Get the API key in personal setting
3. Store API key (a string)to some file as: ~/wandb_api_key_file.txt
4. Install wandb: pip install wandb
5. Fill the "wandb_key_file", "wandb_project" keys in our train function.

Note1: You don't need to specify who own "wandb_project", for example, in team "drivingforce"'s project
"representation", you only need to fill wandb_project="representation"

Note2: In wanbd, there are "team name", "project name", "group name" and "trial_name". We only need to care
"team name" and "project name". The "team name" is set to "drivingforce" by default. You can also use None to
log result to your personal domain. The "group name" of the experiment is exactly the "exp_name" in our context, like
"0304_train_ppo" or so.

Note3: It would be great to change the x-axis in wandb website to "timesteps_total".

Peng Zhenghao, 20210402
"""
from ray import tune

from scenarionet_training.train_utils.utils import train

if __name__ == "__main__":
    config = dict(env="CartPole-v0", num_workers=0, lr=tune.grid_search([1e-2, 1e-4]))
    train(
        "PPO",
        exp_name="test_wandb",
        stop=10000,
        config=config,
        custom_callback=False,
        test_mode=False,
        local_mode=False,
        wandb_project="TEST",
        wandb_team="drivingforce"  # drivingforce is set to default. Use None to log to your personal domain!
    )
