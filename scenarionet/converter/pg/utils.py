from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.scenario.scenario_description import ScenarioDescription as SD


def convert_pg_scenario(scenario_index, version, env):
    """
    Simulate to collect PG Scenarios
    :param scenario_index: the index to export
    :param version: place holder
    :param env: metadrive env instance
    """
    policy = lambda x: [0, 1]  # placeholder
    scenarios, done_info = env.export_scenarios(policy, scenario_index=[scenario_index], to_dict=False)
    scenario = scenarios[scenario_index]
    assert scenario[SD.VERSION] == version, "Data version mismatch"
    return scenario


def get_pg_scenarios(num_scenarios, policy, start_seed=0):
    env = MetaDriveEnv(dict(start_seed=start_seed,
                            num_scenarios=num_scenarios,
                            traffic_density=0.2,
                            agent_policy=policy,
                            crash_vehicle_done=False,
                            map=2))
    return [i for i in range(num_scenarios)], env
