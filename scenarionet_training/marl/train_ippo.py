from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
from ray import tune

from scenarionet_training.marl.algo.ippo import IPPOTrainer
from scenarionet_training.marl.utils.callbacks import MultiAgentDrivingCallbacks
from scenarionet_training.marl.utils.env_wrappers import get_rllib_compatible_env
from scenarionet_training.marl.utils.train import train
from scenarionet_training.marl.utils.utils import get_train_parser

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"

    # Setup config
    stop = int(100_0000)

    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=tune.grid_search([
            # get_rllib_compatible_env(MultiAgentParkingLotEnv),
            get_rllib_compatible_env(MultiAgentIntersectionEnv),
            # get_rllib_compatible_env(MultiAgentTollgateEnv),
            # get_rllib_compatible_env(MultiAgentBottleneckEnv),
            # get_rllib_compatible_env(MultiAgentRoundaboutEnv),
            # get_rllib_compatible_env(MultiAgentMetaDrive),
        ]),
        # env_config=dict(start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]), ),

        # ===== Resource =====
        # So we need 2 CPUs per trial, 0.25 GPU per trial!
        num_gpus=0.25 if args.num_gpus != 0 else 0,

        vf_clip_param=tune.grid_search([10, 20, 50, 100, 1000])
    )

    # Launch training
    train(
        IPPOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        num_seeds=1,
        test_mode=args.test,
        custom_callback=MultiAgentDrivingCallbacks,

        wandb_project="newcopo",
        wandb_team="drivingforce",
        wandb_log_config=False,

        # fail_fast='raise',
        # local_mode=True
    )
