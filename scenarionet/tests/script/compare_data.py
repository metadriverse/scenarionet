from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.scenario.utils import read_scenario_data, read_dataset_summary, assert_scenario_equal
from scenarionet.common_utils import read_scenario

if __name__ == '__main__':
    data_1 = "D:\\code\\scenarionet\\dataset\pg_2000"
    data_2 = "C:\\Users\\x1\\Desktop\\remote"
    summary_1, lookup_1, mapping_1 = read_dataset_summary(data_1)
    summary_2, lookup_2, mapping_2 = read_dataset_summary(data_2)
    # assert lookup_1[-10:] == lookup_2
    scenarios_1 = {}
    scenarios_2 = {}

    for i in range(9):
        scenarios_1[str(i)] = read_scenario(data_1, mapping_1, lookup_1[-9 + i])
        scenarios_2[str(i)] = read_scenario(data_2, mapping_2, lookup_2[i])
    # assert_scenario_equal(scenarios_1, scenarios_2, check_self_type=False, only_compare_sdc=True)
