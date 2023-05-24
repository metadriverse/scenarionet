import os.path

from metadrive.envs.scenario_env import ScenarioEnv

from scenarionet import SCENARIONET_REPO_PATH, SCENARIONET_DATASET_PATH
from scenarionet_training.train.multi_worker_PPO import MultiWorkerPPO
from scenarionet_training.train.utils import train, get_train_parser, get_exp_name

config = dict(
    env=ScenarioEnv,
    env_config=dict(
        # scenario
        start_scenario_index=0,
        num_scenarios=40000,
        data_directory=os.path.join(SCENARIONET_DATASET_PATH, "pg"),
        sequential_seed=True,

        # curriculum training
        curriculum_level=100,
        target_success_rate=0.85,
        # episodes_to_evaluate_curriculum=400,  # default=num_scenarios/curriculum_level

        # traffic & light
        reactive_traffic=False,
        no_static_vehicles=True,
        no_light=True,

        # training
        horizon=None,
        use_lateral_reward=True,
    ),

    # ===== Evaluation =====
    evaluation_interval=15,
    evaluation_num_episodes=1000,
    # 2000 envs each time for efficiency TODO LQY, do eval on all scenarios after training!
    evaluation_config=dict(env_config=dict(start_scenario_index=40000,
                                           num_scenarios=1000,
                                           sequential_seed=True,
                                           curriculum_level=1,  # turn off
                                           data_directory=os.path.join(SCENARIONET_DATASET_PATH, "pg"))),
    evaluation_num_workers=10,
    metrics_smoothing_episodes=10,

    # ===== Training =====
    model=dict(fcnet_hiddens=[512, 256, 128]),
    horizon=600,
    num_sgd_iter=20,
    lr=5e-5,
    rollout_fragment_length=500,
    sgd_minibatch_size=100,
    train_batch_size=40000,
    num_gpus=0,
    num_cpus_per_worker=0.3,
    num_cpus_for_driver=1,
    num_workers=25,
    framework="tf"
)

if __name__ == '__main__':
    # PG data is generated with seeds 10,000 to 60,000
    args = get_train_parser().parse_args()
    exp_name = get_exp_name(args)
    stop = int(100_000_000)
    config["num_gpus"] = 0.5 if args.num_gpus != 0 else 0

    train(
        MultiWorkerPPO,
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
