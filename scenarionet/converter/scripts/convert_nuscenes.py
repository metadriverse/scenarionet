"""
This script aims to convert nuscenes scenarios to ScenarioDescription, so that we can load any nuscenes scenarios into
MetaDrive.
"""

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.nuscenes.utils import convert_one_nuscenes_scenario
from scenarionet.converter.utils import write_to_directory

try:
    from nuscenes import NuScenes
except ImportError:
    print("Can not find nuscenes-devkit")

#
if __name__ == "__main__":
    output_path = SCENARIONET_DATASET_PATH
    version = 'v1.0-mini'
    dataroot = '/home/shady/data/nuscenes'
    force_overwrite = True
    nusc = NuScenes(version=version, dataroot=dataroot)
    scenarios = nusc.scene
    write_to_directory(convert_func=convert_one_nuscenes_scenario,
                       scenarios=scenarios,
                       output_path=output_path,
                       version=version,
                       dataset_name="nuscenes",
                       force_overwrite=True,
                       nuscenes=nusc)
