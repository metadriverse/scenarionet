import os
import os.path

from scenarionet import SCENARIONET_PACKAGE_PATH, TMP_PATH
from scenarionet.builder.utils import merge_database
from scenarionet.common_utils import read_dataset_summary, read_scenario
from scenarionet.verifier.utils import verify_database, set_random_drop


def test_verify_completeness():
    dataset_name = "nuscenes"
    original_dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset", dataset_name)
    dataset_paths = [original_dataset_path + "_{}".format(i) for i in range(5)]

    output_path = os.path.join(TMP_PATH, "combine")
    merge_database(output_path, *dataset_paths, exist_ok=True, overwrite=True, try_generate_missing_file=True)
    dataset_path = output_path
    summary, sorted_scenarios, mapping = read_dataset_summary(dataset_path)
    for scenario_file in sorted_scenarios:
        read_scenario(dataset_path, mapping, scenario_file)
    set_random_drop(True)
    success, result = verify_database(
        dataset_path, error_file_path=TMP_PATH, steps_to_run=0, num_workers=4, overwrite=True
    )
    assert not success

    set_random_drop(False)
    success, result = verify_database(
        dataset_path, error_file_path=TMP_PATH, steps_to_run=0, num_workers=4, overwrite=True
    )
    assert success


if __name__ == '__main__':
    test_verify_completeness()
