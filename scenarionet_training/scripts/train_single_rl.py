import os.path

from metadrive.envs.gymnasium_wrapper import GymnasiumEnvWrapper
from metadrive.envs.scenario_env import ScenarioEnv

from scenarionet import SCENARIONET_REPO_PATH, SCENARIONET_DATASET_PATH
from scenarionet_training.train.utils import train, get_train_parser

if __name__ == '__main__':
    env = GymnasiumEnvWrapper.build(ScenarioEnv)
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "TEST"
    stop = int(100_000_000)

    config = dict(
        env=env,
        env_config=dict(
            # scenario
            num_scenarios=64,
            data_directory=os.path.join(SCENARIONET_DATASET_PATH, "pg"),
            sequential_seed=True,
            # traffic & light
            reactive_traffic=False,
            no_static_vehicles=True,
            no_light=True,
            # curriculum training
            curriculum_level=1,
            episodes_to_evaluate_curriculum=10,
            # training
            horizon=400
        ),

        # ===== Evaluation =====
        evaluation_interval=5,
        evaluation_num_episodes=10,
        evaluation_config=dict(env_config=dict(num_scenarios=10,
                                               sequential_seed=True,
                                               curriculum_level=1,
                                               data_directory=os.path.join(SCENARIONET_DATASET_PATH, "pg"))),
        evaluation_num_workers=1,
        metrics_smoothing_episodes=10,

        # ===== Training =====
        horizon=400,
        num_sgd_iter=20,
        lr=5e-5,
        rollout_fragment_length="auto",
        sgd_minibatch_size=100,
        train_batch_size=20000,
        num_gpus=0.01 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.1,
        num_cpus_for_driver=0.2,
        num_workers=8,
    )

    train(
        "PPO",
        exp_name=exp_name,
        save_dir=os.path.join(SCENARIONET_REPO_PATH, "experiment"),
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=args.num_seeds,
        num_seeds=5,
        test_mode=args.test,
        # local_mode=True,
        # TODO remove this when we release our code
        # wandb_key_file="~/wandb_api_key_file.txt",
        wandb_project="scenarionet",
    )
