"""
This script aims to convert nuplan scenarios to ScenarioDescription, so that we can load any nuplan scenarios into
MetaDrive.
"""
import argparse
import os

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.nuplan.utils import get_nuplan_scenarios, convert_nuplan_scenario
from scenarionet.converter.utils import write_to_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-n", default="nuplan",
                        help="Dataset name, will be used to generate scenario files")
    parser.add_argument("--dataset_path", "-d", default=os.path.join(SCENARIONET_DATASET_PATH, "nuplan"),
                        help="The path of the dataset")
    parser.add_argument("--version", "-v", default='v1.1', help="version")
    parser.add_argument("--overwrite", action="store_true", help="If the dataset_path exists, overwrite it")
    args = parser.parse_args()

    force_overwrite = True
    dataset_name = args.dataset_name
    output_path = args.dataset_path
    version = args.version

    data_root = os.path.join(os.getenv("NUPLAN_DATA_ROOT"), "nuplan-v1.1/splits/mini")
    map_root = os.getenv("NUPLAN_MAPS_ROOT")
    scenarios = get_nuplan_scenarios(data_root, map_root, logs=["2021.07.16.20.45.29_veh-35_01095_01486"])

    write_to_directory(
        convert_func=convert_nuplan_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        force_overwrite=force_overwrite,
        num_workers=8,
    )
