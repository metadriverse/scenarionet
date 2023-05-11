import copy
import os
import os.path

from metadrive.scenario.scenario_description import ScenarioDescription as SD

from scenarionet import SCENARIONET_PACKAGE_PATH, TMP_PATH
from scenarionet.builder.utils import combine_dataset
from scenarionet.common_utils import read_dataset_summary, read_scenario
from scenarionet.common_utils import recursive_equal
from scenarionet.verifier.error import ErrorFile
from scenarionet.verifier.utils import verify_loading_into_metadrive, set_random_drop


def test_generate_from_error():
    set_random_drop(True)
    dataset_name = "nuscenes"
    original_dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset", dataset_name)
    test_dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset")
    dataset_paths = [original_dataset_path + "_{}".format(i) for i in range(5)]
    dataset_path = os.path.join(TMP_PATH, "combine")
    combine_dataset(dataset_path, *dataset_paths, exist_ok=True, try_generate_missing_file=True, overwrite=True)

    summary, sorted_scenarios, mapping = read_dataset_summary(dataset_path)
    for scenario_file in sorted_scenarios:
        read_scenario(dataset_path, mapping, scenario_file)
    success, logs = verify_loading_into_metadrive(
        dataset_path, result_save_dir=test_dataset_path, steps_to_run=1000, num_workers=3
    )
    set_random_drop(False)
    # get error file
    file_name = ErrorFile.get_error_file_name(dataset_path)
    error_file_path = os.path.join(test_dataset_path, file_name)
    # regenerate
    pass_dataset = os.path.join(TMP_PATH, "passed_senarios")
    fail_dataset = os.path.join(TMP_PATH, "failed_scenarios")
    pass_summary, pass_mapping = ErrorFile.generate_dataset(
        error_file_path, pass_dataset, overwrite=True, broken_scenario=False
    )
    fail_summary, fail_mapping = ErrorFile.generate_dataset(
        error_file_path, fail_dataset, overwrite=True, broken_scenario=True
    )

    # assert
    read_pass_summary, _, read_pass_mapping = read_dataset_summary(pass_dataset)
    assert recursive_equal(read_pass_summary, pass_summary)
    assert recursive_equal(read_pass_mapping, pass_mapping)
    read_fail_summary, _, read_fail_mapping, = read_dataset_summary(fail_dataset)
    assert recursive_equal(read_fail_mapping, fail_mapping)
    assert recursive_equal(read_fail_summary, fail_summary)

    # assert pass+fail = origin
    all_summaries = copy.deepcopy(read_pass_summary)
    all_summaries.update(fail_summary)
    assert recursive_equal(all_summaries, summary)

    # test read
    for scenario in read_pass_summary:
        sd = read_scenario(pass_dataset, read_pass_mapping, scenario)
        SD.sanity_check(sd)
    for scenario in read_fail_summary:
        sd = read_scenario(fail_dataset, read_fail_mapping, scenario)
        SD.sanity_check(sd)


if __name__ == '__main__':
    test_generate_from_error()
