"""
This script aims to convert nuscenes scenarios to ScenarioDescription, so that we can load any nuscenes scenarios into
MetaDrive.
"""
import argparse
import os.path

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.nuscenes.utils import convert_nuscenes_scenario, get_nuscenes_scenarios
from scenarionet.converter.utils import write_to_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-n", default="nuscenes",
                        help="Dataset name, will be used to generate scenario files")
    parser.add_argument("--dataset_path", "-d", default=os.path.join(SCENARIONET_DATASET_PATH, "nuscenes"),
                        help="The path of the dataset")
    parser.add_argument("--version", "-v", default='v1.0-mini',  help="version")
    parser.add_argument("--overwrite", action="store_true", help="If the dataset_path exists, overwrite it")
    args = parser.parse_args()

    force_overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.dataset_path
    version = args.version

    dataroot = '/home/shady/data/nuscenes'
    scenarios, nusc = get_nuscenes_scenarios(dataroot, version)

    write_to_directory(
        convert_func=convert_nuscenes_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        force_overwrite=force_overwrite,
        nuscenes=nusc
    )
