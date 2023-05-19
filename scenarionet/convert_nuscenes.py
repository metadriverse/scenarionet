import pkg_resources  # for suppress warning
import argparse
import os.path
from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.nuscenes.utils import convert_nuscenes_scenario, get_nuscenes_scenarios
from scenarionet.converter.utils import write_to_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database_path",
        "-d",
        default=os.path.join(SCENARIONET_DATASET_PATH, "nuscenes"),
        help="directory, The path to place the data"
    )
    parser.add_argument(
        "--dataset_name", "-n", default="nuscenes", help="Dataset name, will be used to generate scenario files"
    )
    parser.add_argument(
        "--version",
        "-v",
        default='v1.0-mini',
        help="version of nuscenes data, scenario of this version will be converted "
    )
    parser.add_argument("--overwrite", action="store_true", help="If the database_path exists, whether to overwrite it")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers to use")
    args = parser.parse_args()

    overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.database_path
    version = args.version

    dataroot = '/home/shady/data/nuscenes'
    scenarios, nuscs = get_nuscenes_scenarios(dataroot, version, args.num_workers)

    write_to_directory(
        convert_func=convert_nuscenes_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        overwrite=overwrite,
        num_workers=args.num_workers,
        nuscenes=nuscs,
    )
