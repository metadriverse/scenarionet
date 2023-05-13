import logging

from metadrive.scenario.scenario_description import ScenarioDescription as SD


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
