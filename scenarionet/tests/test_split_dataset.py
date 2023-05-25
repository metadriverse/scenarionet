import os
import os.path

from scenarionet import SCENARIONET_PACKAGE_PATH, TMP_PATH
from scenarionet.builder.utils import merge_database, split_database
from scenarionet.common_utils import read_dataset_summary


def test_split_dataset():
    dataset_name = "nuscenes"
    original_dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset", dataset_name)
    test_dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset")
    dataset_paths = [original_dataset_path + "_{}".format(i) for i in [0, 1, 3, 4]]

    output_path = os.path.join(TMP_PATH, "combine")
    merge_database(output_path, *dataset_paths, exist_ok=True, overwrite=True, try_generate_missing_file=True)

    #  split
    from_path = output_path
    to_path = os.path.join(TMP_PATH, "split", "split")
    summary_1, lookup_1, mapping_1 = read_dataset_summary(from_path)

    split_database(from_path, to_path, start_index=3, random=True, num_scenarios=4, overwrite=True, exist_ok=True)
    summary_2, lookup_2, mapping_2 = read_dataset_summary(to_path)
    assert len(summary_2) == 4
    for scenario in summary_2:
        assert scenario not in lookup_1[:3]

    split_database(from_path, to_path, start_index=3, num_scenarios=4, overwrite=True, exist_ok=True)
    summary_2, lookup_2, mapping_2 = read_dataset_summary(to_path)
    assert lookup_1[3:7] == lookup_2
    for k in range(4):
        assert summary_1[lookup_2[k]] == summary_2[lookup_2[k]]


if __name__ == '__main__':
    test_split_dataset()
