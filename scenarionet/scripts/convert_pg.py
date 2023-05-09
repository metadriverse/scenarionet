import argparse
import os.path

import metadrive

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.pg.utils import get_pg_scenarios, convert_pg_scenario
from scenarionet.converter.utils import write_to_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", "-n", default="pg", help="Dataset name, will be used to generate scenario files"
    )
    parser.add_argument(
        "--dataset_path", "-d", default=os.path.join(SCENARIONET_DATASET_PATH, "pg"),
        help="directory, The path to place the data"
    )
    parser.add_argument("--version", "-v", default=metadrive.constants.DATA_VERSION, help="version")
    parser.add_argument("--overwrite", action="store_true", help="If the dataset_path exists, overwrite it")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers to use")
    parser.add_argument("--num_scenarios", type=int, default=64, help="how many scenarios to generate (default: 30)")
    parser.add_argument("--start_index", type=int, default=0, help="which index to start")
    args = parser.parse_args()

    force_overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.dataset_path
    version = args.version

    scenario_indices = get_pg_scenarios(args.start_index, args.num_scenarios)

    write_to_directory(
        convert_func=convert_pg_scenario,
        scenarios=scenario_indices,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        force_overwrite=force_overwrite,
        num_workers=args.num_workers,
    )
