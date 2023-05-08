import argparse
import logging
import os

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.utils import write_to_directory
from scenarionet.converter.waymo.utils import convert_waymo_scenario, get_waymo_scenarios

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-n", default="waymo",
                        help="Dataset name, will be used to generate scenario files")
    parser.add_argument("--dataset_path", "-d", default=os.path.join(SCENARIONET_DATASET_PATH, "waymo"),
                        help="The path of the dataset")
    parser.add_argument("--version", "-v", default='v1.2', required=True, help="version")
    parser.add_argument("--overwrite", action="store_true", help="If the dataset_path exists, overwrite it")
    args = parser.parse_args()

    force_overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.dataset_path
    version = args.version

    waymo_data_direction = os.path.join(SCENARIONET_DATASET_PATH, "waymo_origin")
    scenarios = get_waymo_scenarios(waymo_data_direction)

    write_to_directory(
        convert_func=convert_waymo_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        force_overwrite=force_overwrite
    )
