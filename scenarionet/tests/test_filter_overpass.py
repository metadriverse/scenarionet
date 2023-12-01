import os
import os.path

from metadrive.engine.asset_loader import AssetLoader

from scenarionet import SCENARIONET_PACKAGE_PATH, TMP_PATH
from scenarionet.builder.filters import ScenarioFilter
from scenarionet.builder.utils import merge_database


def test_filter_overpass():
    overpass_1 = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset", "overpass")
    overpass_in_md = AssetLoader.file_path("waymo", unix_style=False)
    dataset_paths = [overpass_1, overpass_in_md]

    output_path = os.path.join(TMP_PATH, "combine")
    merge_database(output_path, *dataset_paths, exist_ok=True, overwrite=True, try_generate_missing_file=True)

    #  filter
    filters = []
    filters.append(ScenarioFilter.make(ScenarioFilter.no_overpass))

    summaries, _ = merge_database(
        output_path, *dataset_paths, exist_ok=True, overwrite=True, try_generate_missing_file=True, filters=filters
    )
    assert len(summaries) == 3
    for scenario in summaries:
        assert scenario != "sd_waymo_v1.2_eb4b91b10ca94ff2.pkl"


if __name__ == '__main__':
    test_filter_overpass()
