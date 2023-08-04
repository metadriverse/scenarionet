import os.path

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.envs.gym_wrapper import createGymWrapper

from scenarionet import SCENARIONET_REPO_PATH, SCENARIONET_DATASET_PATH
from scenarionet_training.train_utils.multi_worker_PPO import MultiWorkerPPO
from scenarionet_training.train_utils.utils import train, get_train_parser, get_exp_name

if __name__ == '__main__':
    env = ScenarioEnv
    args = get_train_parser().parse_args()
    exp_name = get_exp_name(args)
    stop = int(100_000_000)

    config = dict(
        env=createGymWrapper(ScenarioEnv),
        env_config=dict(
            # scenario
            start_scenario_index=0,
            num_scenarios=32,
            data_directory=os.path.join(SCENARIONET_DATASET_PATH, "pg"),
            sequential_seed=True,

            # traffic & light
            reactive_traffic=False,
            no_static_vehicles=True,
            no_light=True,
            static_traffic_object=True,

            # curriculum training
            curriculum_level=4,
            target_success_rate=0.8,

            # training
            horizon=None,
        ),

        # # ===== Evaluation =====
        evaluation_interval=2,
        evaluation_num_episodes=32,
        evaluation_config=dict(env_config=dict(start_scenario_index=32,
                                               num_scenarios=32,
                                               sequential_seed=True,
                                               curriculum_level=1,  # turn off
                                               data_directory=os.path.join(SCENARIONET_DATASET_PATH, "pg"))),
        evaluation_num_workers=2,
        metrics_smoothing_episodes=10,

        # ===== Training =====
        model=dict(fcnet_hiddens=[512, 256, 128]),
        horizon=600,
        num_sgd_iter=20,
        lr=5e-5,
        rollout_fragment_length=500,
        sgd_minibatch_size=100,
        train_batch_size=4000,
        num_gpus=0.5 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.4,
        num_cpus_for_driver=1,
        num_workers=2,
        framework="tf"
    )

    train(
        MultiWorkerPPO,
        exp_name=exp_name,
        save_dir=os.path.join(SCENARIONET_REPO_PATH, "experiment"),
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=args.num_seeds,
        num_seeds=1,
        # test_mode=args.test,
        # local_mode=True,
        # TODO remove this when we release our code
        # wandb_key_file="~/wandb_api_key_file.txt",
        wandb_project="scenarionet",
    )
