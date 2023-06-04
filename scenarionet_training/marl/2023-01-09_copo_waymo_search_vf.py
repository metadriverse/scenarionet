import os

from ray import tune

from newcopo.metadrive_scenario.marl_envs.marl_waymo_env import MARLWaymoEnv, WAYMO_DATASET_PATH
from scenarionet_training.marl.algo.copo import CoPOTrainer
from scenarionet_training.marl.utils.callbacks import MultiAgentDrivingCallbacks
from scenarionet_training.marl.utils.env_wrappers import get_rllib_compatible_env
from scenarionet_training.marl.utils.train import train
from scenarionet_training.marl.utils.utils import get_train_parser


from metadrive.envs.marl_envs import MultiAgentParkingLotEnv, MultiAgentRoundaboutEnv, MultiAgentBottleneckEnv, \
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentIntersectionEnv
from ray import tune

from scenarionet_training.marl.algo.copo import CoPOTrainer, USE_CENTRALIZED_CRITIC, USE_DISTRIBUTIONAL_LCF, COUNTERFACTUAL
from scenarionet_training.marl.utils.callbacks import MultiAgentDrivingCallbacks
from scenarionet_training.marl.utils.env_wrappers import get_lcf_env, get_rllib_compatible_env
from scenarionet_training.marl.utils.train import train
from scenarionet_training.marl.utils.utils import get_train_parser



EXP_NAME = os.path.basename(__file__).replace(".py", "")


if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or EXP_NAME

    # Setup config
    stop = int(100_0000)

    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=tune.grid_search([
            get_rllib_compatible_env(get_lcf_env(MARLWaymoEnv)),
        ]),
        env_config=dict(
            dataset_path=WAYMO_DATASET_PATH,
            # start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000]),
            start_seed=tune.grid_search([5000, 6000,]),
            start_case_index=0,
            case_num=1000,

            store_map=True,
            store_map_buffer_size=10,

            neighbours_distance=40,  # TODO: This can be search

        ),

        # ===== Meta SVO =====
        initial_svo_std=0.1,
        # **{USE_DISTRIBUTIONAL_SVO: tune.grid_search([True])},
        svo_lr=1e-4,
        svo_num_iters=5,
        use_global_value=True,
        **{USE_CENTRALIZED_CRITIC: False},

        old_value_loss=tune.grid_search([True, False]),

        # ===== PPO hyper =====
        vf_clip_param=tune.grid_search([10, 50, 100]),

        # ===== Evaluation =====
        evaluation_interval=5,
        evaluation_num_episodes=30,
        evaluation_config=dict(
            env_config=dict(
                start_case_index=1000, case_num=150, sequential_seed=True, store_map=True, store_map_buffer_size=150
            ),
        ),
        evaluation_num_workers=3,
        metrics_smoothing_episodes=150,

        # ===== Resource =====
        # So we need 2 CPUs per trial, 0.25 GPU per trial!
        num_gpus=0.5 if args.num_gpus != 0 else 0,
    )

    # Launch training
    train(
        CoPOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=3,
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
