"""
This script aims to convert nuscenes scenarios to ScenarioDescription, so that we can load any nuscenes scenarios into
MetaDrive.
"""
import os.path

from nuscenes import NuScenes
from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.nuscenes.utils import convert_nuscenes_scenario
from scenarionet.converter.utils import write_to_directory

#
if __name__ == "__main__":
    output_path = os.path.join(SCENARIONET_DATASET_PATH, "nuscenes")
    version = 'v1.0-mini'
    dataroot = '/home/shady/data/nuscenes'
    force_overwrite = True
    nusc = NuScenes(version=version, dataroot=dataroot)
    scenarios = nusc.scene

    write_to_directory(convert_func=convert_nuscenes_scenario,
                       scenarios=scenarios,
                       output_path=output_path,
                       dataset_version=version,
                       dataset_name="nuscenes",
                       force_overwrite=force_overwrite,
                       nuscenes=nusc)
