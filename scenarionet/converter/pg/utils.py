import logging

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.scenario.scenario_description import ScenarioDescription as SD


def make_env(start_index, num_scenarios, extra_config=None):
    config = dict(
        start_seed=start_index,
        num_scenarios=num_scenarios,
        traffic_density=0.15,
        agent_policy=IDMPolicy,
        accident_prob=0.5,
        crash_vehicle_done=False,
        crash_object_done=False,
        store_map=False,
        map=2
    )
    extra_config = extra_config or {}
    config.update(extra_config)
    env = MetaDriveEnv(config)
    return env


def convert_pg_scenario(scenario_index, version, env):
    """
    Simulate to collect PG Scenarios
    :param scenario_index: the index to export [env.start_seed, env.start_seed + num_scenarios_per_worker]
    :param version: place holder
    :param env: metadrive env instance
    """
    #
    # if (scenario_index - env.config["start_seed"]) % reset_freq == 0:
    #     # for avoiding memory leak
    #     env.close()

    logging.disable(logging.INFO)
    policy = lambda x: [0, 1]  # placeholder
    scenarios, done_info = env.export_scenarios(
        policy, scenario_index=[scenario_index], max_episode_length=500, suppress_warning=True, to_dict=False
    )
    scenario = scenarios[scenario_index]
    assert scenario[SD.VERSION] == version, "Data version mismatch"
    return scenario


def get_pg_scenarios(start_index, num_scenarios):
    return [i for i in range(start_index, start_index + num_scenarios)]
