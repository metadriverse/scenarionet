import os

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.scenario.utils import get_number_of_scenarios, assert_scenario_equal

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.common_utils import read_dataset_summary, read_scenario

if __name__ == '__main__':

    dataset_path = os.path.abspath(os.path.join(SCENARIONET_DATASET_PATH, "pg"))
    start_seed = 0
    num_scenario = get_number_of_scenarios(dataset_path)

    # load multi process ret
    summary, s_list, mapping = read_dataset_summary(dataset_path)
    to_compare = dict()
    for k, file in enumerate(s_list[:num_scenario]):
        to_compare[k + start_seed] = read_scenario(dataset_path, mapping, file).to_dict()

    # generate single process ret
    env = MetaDriveEnv(
        dict(
            start_seed=start_seed,
            num_scenarios=num_scenario,
            traffic_density=0.15,
            agent_policy=IDMPolicy,
            crash_vehicle_done=False,
            map=2
        )
    )
    policy = lambda x: [0, 1]  # placeholder
    ret = env.export_scenarios(
        policy, [i for i in range(start_seed, start_seed + num_scenario)], return_done_info=False
    )

    # for i in tqdm.tqdm(range(num_scenario), desc="Assert"):
    assert_scenario_equal(ret, to_compare, only_compare_sdc=True)
