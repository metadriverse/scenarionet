import os
import os.path

from scenarionet import SCENARIONET_PACKAGE_PATH
from scenarionet.builder.utils import combine_multiple_dataset, read_dataset_summary, read_scenario
from scenarionet.verifier.utils import verify_loading_into_metadrive


def test_combine_multiple_dataset():
    dataset_name = "nuscenes"
    original_dataset_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset", dataset_name)
    dataset_paths = [original_dataset_path + "_{}".format(i) for i in range(5)]

    output_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "combine")
    combine_multiple_dataset(output_path,
                             *dataset_paths,
                             force_overwrite=True,
                             try_generate_missing_file=True)
    dataset_paths.append(output_path)
    for dataset_path in dataset_paths:
        summary, sorted_scenarios, mapping = read_dataset_summary(dataset_path)
        for scenario_file in sorted_scenarios:
            read_scenario(os.path.join(dataset_path, mapping[scenario_file], scenario_file))
        success, result = verify_loading_into_metadrive(dataset_path,
                                                        result_save_dir="./test_dataset",
                                                        steps_to_run=300)
        assert success


if __name__ == '__main__':
    test_combine_multiple_dataset()
