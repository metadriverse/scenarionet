import os
import os.path
from metadrive.scenario.utils import assert_scenario_equal
from scenarionet import SCENARIONET_PACKAGE_PATH
from scenarionet.builder.utils import combine_multiple_dataset
from scenarionet.common_utils import read_dataset_summary, read_scenario
from scenarionet.verifier.error import ErrorFile
from scenarionet.verifier.utils import verify_loading_into_metadrive, set_random_drop


def test_combine_multiple_dataset():
    set_random_drop(True)
    dataset_name = "nuscenes"
    original_dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset", dataset_name)
    dataset_paths = [original_dataset_path + "_{}".format(i) for i in range(5)]
    dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "combine")
    combine_multiple_dataset(dataset_path, *dataset_paths, force_overwrite=True, try_generate_missing_file=True)

    summary, sorted_scenarios, mapping = read_dataset_summary(dataset_path)
    for scenario_file in sorted_scenarios:
        read_scenario(os.path.join(dataset_path, mapping[scenario_file], scenario_file))
    success, logs = verify_loading_into_metadrive(
        dataset_path, result_save_dir="test_dataset", steps_to_run=1000, num_workers=4)
    set_random_drop(False)
    # regenerate
    file_name = ErrorFile.get_error_file_name(dataset_path)
    error_file_path = os.path.join("test_dataset", file_name)

    pass_dataset = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "passed_senarios")
    fail_dataset = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "failed_scenarios")
    pass_summary, pass_mapping = ErrorFile.generate_dataset(error_file_path, pass_dataset, broken_scenario=False)
    fail_summary, fail_mapping = ErrorFile.generate_dataset(error_file_path, fail_dataset, broken_scenario=True)

    read_pass_summary, _, read_pass_mapping = read_dataset_summary(pass_dataset)
    read_fail_summary, _, read_fail_mapping, = read_dataset_summary(fail_dataset)



if __name__ == '__main__':
    test_combine_multiple_dataset()
