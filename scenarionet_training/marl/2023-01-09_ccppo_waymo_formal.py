import os

from ray import tune

from newcopo.metadrive_scenario.marl_envs.marl_waymo_env import MARLWaymoEnv, WAYMO_DATASET_PATH
from scenarionet_training.marl.algo.ccppo import CCPPOTrainer, get_ccppo_env
from scenarionet_training.marl.utils.callbacks import MultiAgentDrivingCallbacks
from scenarionet_training.marl.utils.train import train
from scenarionet_training.marl.utils.utils import get_train_parser

EXP_NAME = os.path.basename(__file__).replace(".py", "")

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or EXP_NAME

    stop = int(100_0000)
    config = dict(
        # ===== Environmental Setting =====
        env=tune.grid_search([get_ccppo_env(MARLWaymoEnv)]),
        env_config=dict(
            dataset_path=WAYMO_DATASET_PATH,
            # start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000]),
            start_seed=tune.grid_search([5000, 6000, 7000]),
            start_case_index=0,
            case_num=1000,

            store_map=True,
            store_map_buffer_size=10,

            neighbours_distance=40,  # TODO: This can be search

        ),

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
        num_gpus=0.25 if args.num_gpus != 0 else 0,

        # ===== CCPPO =====
        counterfactual=True,
        fuse_mode=tune.grid_search(["mf", "concat"]),
        mf_nei_distance=10,

        # ===== PPO =====
        vf_clip_param=100,
        old_value_loss=True
    )

    # Launch training
    train(
        CCPPOTrainer,
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
