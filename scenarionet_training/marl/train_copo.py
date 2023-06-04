from metadrive.envs.marl_envs import MultiAgentParkingLotEnv, MultiAgentRoundaboutEnv, MultiAgentBottleneckEnv, \
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentIntersectionEnv
from ray import tune

from scenarionet_training.marl.algo.copo import CoPOTrainer, USE_CENTRALIZED_CRITIC, USE_DISTRIBUTIONAL_LCF, COUNTERFACTUAL
from scenarionet_training.marl.utils.callbacks import MultiAgentDrivingCallbacks
from scenarionet_training.marl.utils.env_wrappers import get_lcf_env, get_rllib_compatible_env
from scenarionet_training.marl.utils.train import train
from scenarionet_training.marl.utils.utils import get_train_parser

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"

    # Setup config
    # We set the stop criterion to 2M environmental steps! Since PPO in single OurEnvironment converges at around 20k steps.
    stop = int(100_0000)

    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=tune.grid_search([
            # get_rllib_compatible_env(get_lcf_env(MultiAgentParkingLotEnv)),
            # get_rllib_compatible_env(get_lcf_env(MultiAgentRoundaboutEnv)),
            # get_rllib_compatible_env(get_lcf_env(MultiAgentTollgateEnv)),
            # get_rllib_compatible_env(get_lcf_env(MultiAgentBottleneckEnv)),
            get_rllib_compatible_env(get_lcf_env(MultiAgentIntersectionEnv)),
            # get_rllib_compatible_env(get_lcf_env(MultiAgentMetaDrive)),
        ]),
        env_config=dict(
            # start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]),
            neighbours_distance=40,
        ),

        # ===== Resource =====
        # So we need 0.2(num_cpus_per_worker) * 5(num_workers) + 1(num_cpus_for_driver) = 2 CPUs per trial!
        num_gpus=0.5 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.1,

        # ===== Meta SVO =====
        initial_svo_std=0.1,
        # **{USE_DISTRIBUTIONAL_SVO: tune.grid_search([True])},
        svo_lr=1e-4,
        svo_num_iters=5,
        use_global_value=True,
        **{USE_CENTRALIZED_CRITIC: tune.grid_search([False])},
    )

    # Launch training
    train(
        CoPOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,  # Don't call get_ippo_config here!
        num_gpus=args.num_gpus,
        num_seeds=1,
        # test_mode=True,
        custom_callback=MultiAgentDrivingCallbacks,

        wandb_project="newcopo",
        wandb_team="drivingforce",
        wandb_log_config=False,

        # fail_fast='raise',
        # local_mode=True
    )
