"""
This script aims to convert nuscenes scenarios to ScenarioDescription, so that we can load any nuscenes scenarios into
MetaDrive.
"""
import os.path

try:
    from nuscenes import NuScenes
except ImportError:
    raise ImportError("nuscenes-devkit has to be set up before running data conversion")
from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.nuscenes.utils import convert_nuscenes_scenario
from scenarionet.converter.utils import write_to_directory

if __name__ == "__main__":
    dataset_name = "nuscenes"
    output_path = os.path.join(SCENARIONET_DATASET_PATH, dataset_name)
    version = 'v1.0-mini'
    dataroot = '/home/shady/data/nuscenes'
    force_overwrite = True
    nusc = NuScenes(version=version, dataroot=dataroot)
    scenarios = nusc.scene

    write_to_directory(convert_func=convert_nuscenes_scenario,
                       scenarios=scenarios,
                       output_path=output_path,
                       dataset_version=version,
                       dataset_name=dataset_name,
                       force_overwrite=force_overwrite,
                       nuscenes=nusc)
