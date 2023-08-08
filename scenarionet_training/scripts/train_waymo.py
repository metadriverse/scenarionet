import os.path
from metadrive.envs.gym_wrapper import createGymWrapper
from metadrive.envs.scenario_env import ScenarioEnv
from scenarionet import SCENARIONET_REPO_PATH, SCENARIONET_DATASET_PATH
from scenarionet_training.train_utils.multi_worker_PPO import MultiWorkerPPO
from scenarionet_training.train_utils.utils import train, get_train_parser, get_exp_name

config = dict(
    env=createGymWrapper(ScenarioEnv),
    env_config=dict(
        # scenario
        start_scenario_index=0,
        num_scenarios=40000,
        data_directory=os.path.join(SCENARIONET_DATASET_PATH, "waymo_train"),
        sequential_seed=True,

        # curriculum training
        curriculum_level=100,
        target_success_rate=0.8,
        # episodes_to_evaluate_curriculum=400,  # default=num_scenarios/curriculum_level

        # traffic & light
        reactive_traffic=True,
        no_static_vehicles=True,
        no_light=True,
        static_traffic_object=True,

        # training scheme
        horizon=None,
        driving_reward=1,
        steering_range_penalty=0,
        heading_penalty=1,
        lateral_penalty=1.0,
        no_negative_reward=True,
        on_lane_line_penalty=0,
        crash_vehicle_penalty=2,
        crash_human_penalty=2,
        out_of_road_penalty=2,
        max_lateral_dist=2,
        # crash_vehicle_done=True,

        vehicle_config=dict(side_detector=dict(num_lasers=0))

    ),

                    # ===== Evaluation =====
                    evaluation_interval=15,
                    evaluation_num_episodes=1000,
                    # TODO (LQY), this is a sample from testset do eval on all scenarios after training!
                    evaluation_config=dict(env_config=dict(start_scenario_index=0,
                                                           num_scenarios=1000,
                                                           sequential_seed=True,
                                                           curriculum_level=1,  # turn off
                                                           data_directory=os.path.join(SCENARIONET_DATASET_PATH,
                                                                                       "waymo_test"))),
                    evaluation_num_workers=10,
                    metrics_smoothing_episodes=10,

                    # ===== Training =====
                    model=dict(fcnet_hiddens=[512, 256, 128]),
                    horizon=600,
                    num_sgd_iter=20,
                    lr=1e-4,
                    rollout_fragment_length=500,
                    sgd_minibatch_size=200,
                    train_batch_size=50000,
                    num_gpus=0.5,
                    num_cpus_per_worker=0.3,
                    num_cpus_for_driver=1,
                    num_workers=20,
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
