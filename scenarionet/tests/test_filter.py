import os
import os.path

from metadrive.type import MetaDriveType

from scenarionet import SCENARIONET_PACKAGE_PATH
from scenarionet.builder.filters import ScenarioFilter
from scenarionet.builder.utils import combine_multiple_dataset


def test_filter_dataset():
    dataset_name = "nuscenes"
    original_dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset", dataset_name)
    dataset_paths = [original_dataset_path + "_{}".format(i) for i in range(5)]

    output_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "combine")

    # test 1
    # nuscenes data has no light
    # light_condition = ScenarioFilter.make(ScenarioFilter.has_traffic_light)
    sdc_driving_condition = ScenarioFilter.make(ScenarioFilter.sdc_moving_dist,
                                                target_dist=30,
                                                condition="smaller")
    answer = ['scene-0553', 'scene-0757', 'scene-1100']
    summary, mapping = combine_multiple_dataset(output_path,
                                                *dataset_paths,
                                                force_overwrite=True,
                                                try_generate_missing_file=True,
                                                filters=[sdc_driving_condition])
    assert len(answer) == len(summary)
    for a in answer:
        in_ = False
        for s in summary:
            if a in s:
                in_ = True
                break
        assert in_

    # test 2
    num_condition = ScenarioFilter.make(ScenarioFilter.object_number,
                                        number_threshold=6,
                                        object_type=MetaDriveType.PEDESTRIAN,
                                        condition="greater")


if __name__ == '__main__':
    test_filter_dataset()
