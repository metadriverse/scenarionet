import pkg_resources  # for suppress warning
import argparse
import os
from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.nuplan.utils import get_nuplan_scenarios, convert_nuplan_scenario
from scenarionet.converter.utils import write_to_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", "-n", default="nuplan", help="Dataset name, will be used to generate scenario files"
    )
    parser.add_argument(
        "--dataset_path",
        "-d",
        default=os.path.join(SCENARIONET_DATASET_PATH, "nuplan"),
        help="A directory, the path to place the data"
    )
    parser.add_argument("--version", "-v", default='v1.1', help="version of the raw data")
    parser.add_argument("--overwrite", action="store_true", help="If the dataset_path exists, whether to overwrite it")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers to use")
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default=os.path.join(os.getenv("NUPLAN_DATA_ROOT"), "nuplan-v1.1/splits/mini"),
        help="the place store .db files"
    )
    parser.add_argument("--test", action="store_true", help="for test use only. convert one log")
    args = parser.parse_args()

    overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.dataset_path
    version = args.version

    data_root = args.raw_data_path
    map_root = os.getenv("NUPLAN_MAPS_ROOT")
    if args.test:
        scenarios = get_nuplan_scenarios(data_root, map_root, logs=["2021.07.16.20.45.29_veh-35_01095_01486"])
    else:
        scenarios = get_nuplan_scenarios(data_root, map_root)

    write_to_directory(
        convert_func=convert_nuplan_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        overwrite=overwrite,
        num_workers=args.num_workers
    )
