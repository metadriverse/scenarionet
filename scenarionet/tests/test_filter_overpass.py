import os
import os.path

from scenarionet import SCENARIONET_PACKAGE_PATH, TMP_PATH
from scenarionet.builder.utils import merge_database, split_database
from scenarionet.common_utils import read_dataset_summary
import argparse

from scenarionet.builder.filters import ScenarioFilter
from scenarionet.builder.utils import merge_database


def test_filter_overpass():
    dataset_name = "nuscenes"
    original_dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset", dataset_name)
    overpass = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset", "overpass")
    dataset_paths = [original_dataset_path + "_{}".format(i) for i in [0, 1, 3, 4]]
    dataset_paths.append(overpass)

    output_path = os.path.join(TMP_PATH, "combine")
    merge_database(output_path, *dataset_paths, exist_ok=True, overwrite=True, try_generate_missing_file=True)

    #  filter
    filters = []
    filters.append(ScenarioFilter.make(ScenarioFilter.no_overpass))

    summaries, _ = merge_database(
        output_path,
        *dataset_paths,
        exist_ok=True,
        overwrite=True,
        try_generate_missing_file=True,
        filters=filters
    )
    for scenario in summaries:
        assert "waymo" not in scenario


if __name__ == '__main__':
    test_filter_overpass()
