import argparse

from scenarionet.builder.utils import combine_multiple_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", required=True, help="Dataset path, a directory")
    parser.add_argument('--from_datasets', required=True, nargs='+', default=[])
    parser.add_argument("--overwrite", action="store_true", help="If the dataset_path exists, overwrite it")
    args = parser.parse_args()
    if len(args.from_datasets) != 0:
        combine_multiple_dataset(
            args.to, *args.from_datasets, force_overwrite=args.overwrite, try_generate_missing_file=True
        )
