import os.path
from ray.tune import grid_search
from metadrive.envs.gymnasium_wrapper import GymnasiumEnvWrapper
from metadrive.envs.scenario_env import ScenarioEnv

from scenarionet import SCENARIONET_REPO_PATH, SCENARIONET_DATASET_PATH
from scenarionet_training.train.utils import train, get_train_parser, get_exp_name

if __name__ == '__main__':
    env = ScenarioEnv
    args = get_train_parser().parse_args()
    exp_name = get_exp_name(args)
    stop = int(100_000_000)

    config = dict(
        env=env,
        env_config=dict(
            # scenario
            start_scenario_index=0,
            num_scenarios=8000,
            data_directory=os.path.join(SCENARIONET_DATASET_PATH, "pg"),
            sequential_seed=True,

            # traffic & light
            reactive_traffic=False,
            no_static_vehicles=True,
            no_light=True,

            # curriculum training
            curriculum_level=80,
            episodes_to_evaluate_curriculum=50,
            target_success_rate=0.8,

            # training
            horizon=400
        ),

        # ===== Evaluation =====
        evaluation_interval=10,
        evaluation_num_episodes=100,
        evaluation_config=dict(env_config=dict(start_scenario_index=8000,
                                               num_scenarios=2000,
                                               sequential_seed=False,
                                               curriculum_level=1, # turn off
                                               data_directory=os.path.join(SCENARIONET_DATASET_PATH, "pg"))),
        evaluation_num_workers=2,
        metrics_smoothing_episodes=50,

        # ===== Training =====
        model=dict(fcnet_hiddens=[512, 256, 128]),
        horizon=400,
        num_sgd_iter=20,
        lr=5e-5,
        rollout_fragment_length=500,
        sgd_minibatch_size=100,
        train_batch_size=40000,
        num_gpus=0.5 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.4,
        num_cpus_for_driver=1,
        num_workers=10,
        framework="tf"
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
