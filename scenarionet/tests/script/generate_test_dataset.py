"""
This script aims to convert nuscenes scenarios to ScenarioDescription, so that we can load any nuscenes scenarios into
MetaDrive.
"""
import os.path

from scenarionet import SCENARIONET_PACKAGE_PATH
from scenarionet.converter.nuscenes.utils import convert_nuscenes_scenario, get_nuscenes_scenarios
from scenarionet.converter.utils import write_to_directory

if __name__ == "__main__":
    dataset_name = "nuscenes"
    output_path = os.path.join(SCENARIONET_PACKAGE_PATH, "tests", "test_dataset", dataset_name)
    version = 'v1.0-mini'
    force_overwrite = True

    dataroot = '/home/shady/data/nuscenes'
    scenarios, nusc = get_nuscenes_scenarios(dataroot, version)

    for i in range(5):
        write_to_directory(
            convert_func=convert_nuscenes_scenario,
            scenarios=scenarios[i * 2:i * 2 + 2],
            output_path=output_path + "_{}".format(i),
            dataset_version=version,
            dataset_name=dataset_name,
            force_overwrite=force_overwrite,
            nuscenes=nusc)
