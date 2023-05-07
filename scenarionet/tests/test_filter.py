import os
import os.path
from scenarionet.builder.filters import ScenarioFilter
from scenarionet import SCENARIONET_PACKAGE_PATH
from scenarionet.builder.utils import combine_multiple_dataset, read_dataset_summary, read_scenario
from scenarionet.verifier.utils import verify_loading_into_metadrive
from metadrive.type import MetaDriveType


def test_filter_dataset():
    dataset_name = "nuscenes"
    original_dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "_test_dataset", dataset_name)
    dataset_paths = [original_dataset_path + "_{}".format(i) for i in range(5)]

    output_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "combine")

    num_condition = ScenarioFilter.make(ScenarioFilter.object_number,
                                        number_threshold=6,
                                        object_type=MetaDriveType.PEDESTRIAN,
                                        condition="greater")
    # nuscenes data has no light
    # light_condition = ScenarioFilter.make(ScenarioFilter.has_traffic_light)
    sdc_driving_condition = ScenarioFilter.make(ScenarioFilter.sdc_moving_dist,
                                                target_dist=2,
                                                condition="smaller")

    summary, mapping = combine_multiple_dataset(output_path,
                                                *dataset_paths,
                                                force_overwrite=True,
                                                try_generate_missing_file=True,
                                                filters=[num_condition, sdc_driving_condition])


if __name__ == '__main__':
    test_filter_dataset()
