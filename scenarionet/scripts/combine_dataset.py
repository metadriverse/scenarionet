import argparse

from scenarionet.builder.filters import ScenarioFilter
from scenarionet.builder.utils import combine_multiple_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", required=True, help="Dataset path, a directory")
    parser.add_argument('--from_datasets', required=True, nargs='+', default=[])
    parser.add_argument("--overwrite", action="store_true", help="If the dataset_path exists, overwrite it")
    parser.add_argument("--sdc_moving_dist_min", default=0, help="Selecting case with sdc_moving_dist > this value")
    args = parser.parse_args()
    filters = [ScenarioFilter.make(ScenarioFilter.sdc_moving_dist, target_dist=20, condition="greater")]

    if len(args.from_datasets) != 0:
        combine_multiple_dataset(
            args.to,
            *args.from_datasets,
            force_overwrite=args.overwrite,
            try_generate_missing_file=True,
            filters=filters
        )
