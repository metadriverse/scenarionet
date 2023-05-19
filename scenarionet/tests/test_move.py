import os
import os.path

import pytest

from scenarionet import SCENARIONET_PACKAGE_PATH, TMP_PATH
from scenarionet.builder.utils import move_database, merge_database
from scenarionet.common_utils import read_dataset_summary, read_scenario
from scenarionet.verifier.utils import verify_database


@pytest.mark.order("first")
def test_move_database():
    dataset_name = "nuscenes"
    original_dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset", dataset_name)
    dataset_paths = [original_dataset_path + "_{}".format(i) for i in range(5)]
    moved_path = []
    output_path = os.path.join(TMP_PATH, "move_combine")
    # move
    for k, from_path in enumerate(dataset_paths):
        to = os.path.join(TMP_PATH, str(k))
        move_database(from_path, to)
        moved_path.append(to)
        # assert not os.path.exists(path)
    merge_database(output_path, *moved_path, exist_ok=True, overwrite=True, try_generate_missing_file=True)
    # verify
    summary, sorted_scenarios, mapping = read_dataset_summary(output_path)
    for scenario_file in sorted_scenarios:
        read_scenario(output_path, mapping, scenario_file)
    success, result = verify_database(
        output_path, result_save_dir=output_path, steps_to_run=0, num_workers=4, overwrite=True
    )
    assert success

    # move 2
    new_move_pathes = []
    for k, from_path in enumerate(moved_path):
        new_p = os.path.join(TMP_PATH, str(k) + str(k))
        new_move_pathes.append(new_p)
        move_database(from_path, new_p, exist_ok=True, overwrite=True)
        assert not os.path.exists(from_path)
    merge_database(output_path, *new_move_pathes, exist_ok=True, overwrite=True, try_generate_missing_file=True)
    # verify
    summary, sorted_scenarios, mapping = read_dataset_summary(output_path)
    for scenario_file in sorted_scenarios:
        read_scenario(output_path, mapping, scenario_file)
    success, result = verify_database(
        output_path, result_save_dir=output_path, steps_to_run=0, num_workers=4, overwrite=True
    )
    assert success


if __name__ == '__main__':
    test_move_database()
