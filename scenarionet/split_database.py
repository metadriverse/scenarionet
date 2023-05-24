"""
This script is for extracting a subset of data from an existing database
"""

import argparse

from scenarionet.builder.utils import split_database

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', required=True, help="Which database to extract data from.")
    parser.add_argument(
        "--to",
        required=True,
        help="The name of the new database. "
        "It will create a new directory to store dataset_summary.pkl and dataset_mapping.pkl. "
        "If exists_ok=True, those two .pkl files will be stored in an existing directory and turn "
        "that directory into a database."
    )
    parser.add_argument("--num_scenarios", type=int, default=64, help="how many scenarios to extract (default: 30)")
    parser.add_argument("--start_index", type=int, default=0, help="which index to start")
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="Still allow to write, if the to_folder exists already. "
        "This write will only create two .pkl files and this directory will become a database."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="When exists ok is set but summary.pkl and map.pkl exists in existing dir, "
        "whether to overwrite both files"
    )
    args = parser.parse_args()
    from_path = args.__getattr__("from")
    split_database(
        from_path,
        args.to,
        args.start_index,
        args.num_scenarios,
        exist_ok=args.exist_ok,
        overwrite=args.overwrite,
    )


