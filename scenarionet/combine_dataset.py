import pkg_resources  # for suppress warning
import argparse
from scenarionet.builder.filters import ScenarioFilter
from scenarionet.builder.utils import combine_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="The name of the new combined dataset. "
        "It will create a new directory to store dataset_summary.pkl and dataset_mapping.pkl. "
        "If exists_ok=True, those two .pkl files will be stored in an existing directory and turn "
        "that directory into a dataset."
    )
    parser.add_argument(
        '--from_datasets',
        required=True,
        nargs='+',
        default=[],
        help="Which datasets to combine. It takes any number of directory path as input"
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="Still allow to write, if the dir exists already. "
        "This write will only create two .pkl files and this directory will become a dataset."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="When exists ok is set but summary.pkl and map.pkl exists in existing dir, "
        "whether to overwrite both files"
    )
    parser.add_argument(
        "--sdc_moving_dist_min",
        default=20,
        help="Selecting case with sdc_moving_dist > this value. "
        "We will add more filter conditions in the future."
    )
    args = parser.parse_args()
    target = args.sdc_moving_dist_min
    filters = [ScenarioFilter.make(ScenarioFilter.sdc_moving_dist, target_dist=target, condition="greater")]

    if len(args.from_datasets) != 0:
        combine_dataset(
            args.dataset_path,
            *args.from_datasets,
            exist_ok=args.exist_ok,
            overwrite=args.overwrite,
            try_generate_missing_file=True,
            filters=filters
        )
    else:
        raise ValueError("No source dataset are provided. Abort.")
