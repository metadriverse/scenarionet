import pkg_resources  # for suppress warning
import shutil
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
    parser.add_argument(
        "--start_file_index",
        default=0,
        type=int,
        help="Control how many files to use. We will list all files in the raw data folder "
             "and select files[start_file_index: start_file_index+num_files]"
    )
    parser.add_argument(
        "--num_files",
        default=1000,
        type=int,
        help="Control how many files to use. We will list all files in the raw data folder "
             "and select files[start_file_index: start_file_index+num_files]"
    )
    args = parser.parse_args()

    overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.database_path
    version = args.version

    save_path = output_path
    if os.path.exists(output_path):
        if not overwrite:
            raise ValueError(
                "Directory {} already exists! Abort. "
                "\n Try setting overwrite=True or adding --overwrite".format(output_path)
            )
        else:
            shutil.rmtree(output_path)

    waymo_data_directory = os.path.join(SCENARIONET_DATASET_PATH, args.raw_data_path)
    scenarios = get_waymo_scenarios(waymo_data_directory, args.start_file_index, args.num_files,
                                    num_workers=8)  # do not use too much worker to read data

    write_to_directory(
        convert_func=convert_waymo_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        overwrite=overwrite,
        num_workers=args.num_workers
    )
