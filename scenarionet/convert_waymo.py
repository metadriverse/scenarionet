import pkg_resources  # for suppress warning
import argparse
import logging
import os

from scenarionet import SCENARIONET_DATASET_PATH, SCENARIONET_REPO_PATH
from scenarionet.converter.utils import write_to_directory
from scenarionet.converter.waymo.utils import convert_waymo_scenario, get_waymo_scenarios

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database_path",
        "-d",
        default=os.path.join(SCENARIONET_DATASET_PATH, "waymo"),
        help="A directory, the path to place the converted data"
    )
    parser.add_argument(
        "--dataset_name", "-n", default="waymo", help="Dataset name, will be used to generate scenario files"
    )
    parser.add_argument("--version", "-v", default='v1.2', help="version")
    parser.add_argument("--overwrite", action="store_true", help="If the database_path exists, whether to overwrite it")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers to use")
    parser.add_argument(
        "--raw_data_path",
        default=os.path.join(SCENARIONET_REPO_PATH, "waymo_origin"),
        help="The directory stores all waymo tfrecord"
    )
    args = parser.parse_args()

    overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.database_path
    version = args.version

    waymo_data_directory = os.path.join(SCENARIONET_DATASET_PATH, args.raw_data_path)
    scenarios = get_waymo_scenarios(waymo_data_directory)

    write_to_directory(
        convert_func=convert_waymo_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        overwrite=overwrite,
        num_workers=args.num_workers
    )
