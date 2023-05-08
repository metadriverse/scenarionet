"""
This script aims to convert nuscenes scenarios to ScenarioDescription, so that we can load any nuscenes scenarios into
MetaDrive.
"""
import argparse
import os.path

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.nuscenes.utils import convert_nuscenes_scenario, get_nuscenes_scenarios
from scenarionet.converter.utils import write_to_directory
from scenarionet.builder.utils import combine_multiple_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--to_dataset", required=True, help="Dataset path, a directory")
    parser.add_argument('--from_datasets', required=True, nargs='+', default=[])
    parser.add_argument("--overwrite", action="store_true", help="If the dataset_path exists, overwrite it")
    args = parser.parse_args()
    if len(args.from_datasets) != 0:
        combine_multiple_dataset(args.dataset_path,
                                 *args.from_dataset,
                                 force_overwrite=args.overwrite,
                                 try_generate_missing_file=True)
