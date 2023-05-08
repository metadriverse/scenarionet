import os

from metadrive.scenario.scenario_description import ScenarioDescription as SD

from scenarionet import SCENARIONET_DATASET_PATH, SCENARIONET_PACKAGE_PATH
from scenarionet.builder.utils import combine_dataset
from scenarionet.common_utils import read_dataset_summary, read_scenario


def _test_combine_dataset():
    dataset_paths = [
        os.path.join(SCENARIONET_DATASET_PATH, "nuscenes"),
        os.path.join(SCENARIONET_DATASET_PATH, "nuscenes", "nuscenes_0"),
        os.path.join(SCENARIONET_DATASET_PATH, "nuplan"),
        os.path.join(SCENARIONET_DATASET_PATH, "waymo"),
        os.path.join(SCENARIONET_DATASET_PATH, "pg")
    ]

    combine_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "tmp", "combine")
    combine_dataset(combine_path, *dataset_paths, exist_ok=True, force_overwrite=True, try_generate_missing_file=True)
    summary, _, mapping = read_dataset_summary(combine_path)
    for scenario in summary:
        sd = read_scenario(combine_path, mapping, scenario)
        SD.sanity_check(sd)
    print("Test pass")


if __name__ == '__main__':
    _test_combine_dataset()
