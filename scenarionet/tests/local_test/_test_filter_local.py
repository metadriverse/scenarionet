import os
import os.path

from metadrive.type import MetaDriveType

from scenarionet import SCENARIONET_DATASET_PATH, SCENARIONET_PACKAGE_PATH, TMP_PATH
from scenarionet.builder.filters import ScenarioFilter
from scenarionet.builder.utils import merge_database


def test_filter_dataset():
    """
    It is just a runnable test
    """
    dataset_paths = [os.path.join(SCENARIONET_DATASET_PATH, "nuscenes")]
    dataset_paths.append(os.path.join(SCENARIONET_DATASET_PATH, "nuplan"))
    dataset_paths.append(os.path.join(SCENARIONET_DATASET_PATH, "waymo"))
    dataset_paths.append(os.path.join(SCENARIONET_DATASET_PATH, "pg"))

    output_path = os.path.join(TMP_PATH, "combine")

    # ========================= test 1 =========================
    # nuscenes data has no light
    # light_condition = ScenarioFilter.make(ScenarioFilter.has_traffic_light)
    sdc_driving_condition = ScenarioFilter.make(ScenarioFilter.sdc_moving_dist, target_dist=30, condition="greater")
    summary, mapping = merge_database(
        output_path,
        *dataset_paths,
        exist_ok=True,
        overwrite=True,
        try_generate_missing_file=True,
        filters=[sdc_driving_condition]
    )
    assert len(summary) > 0

    # ========================= test 2 =========================

    num_condition = ScenarioFilter.make(
        ScenarioFilter.object_number, number_threshold=50, object_type=MetaDriveType.PEDESTRIAN, condition="greater"
    )

    summary, mapping = merge_database(
        output_path,
        *dataset_paths,
        exist_ok=True,
        overwrite=True,
        try_generate_missing_file=True,
        filters=[num_condition]
    )
    assert len(summary) > 0

    # ========================= test 3 =========================

    traffic_light = ScenarioFilter.make(ScenarioFilter.has_traffic_light)

    summary, mapping = merge_database(
        output_path,
        *dataset_paths,
        exist_ok=True,
        overwrite=True,
        try_generate_missing_file=True,
        filters=[traffic_light]
    )
    assert len(summary) > 0


if __name__ == '__main__':
    test_filter_dataset()
