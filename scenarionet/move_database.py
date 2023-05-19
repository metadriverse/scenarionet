import argparse

from scenarionet.builder.utils import move_database

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', required=True, help="Which dataset to move.")
    parser.add_argument(
        "--to",
        required=True,
        help="The name of the new dataset. "
        "It will create a new directory to store dataset_summary.pkl and dataset_mapping.pkl. "
        "If exists_ok=True, those two .pkl files will be stored in an existing directory and turn "
        "that directory into a dataset."
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="Still allow to write, if the to_folder exists already. "
        "This write will only create two .pkl files and this directory will become a dataset."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="When exists ok is set but summary.pkl and map.pkl exists in existing dir, "
        "whether to overwrite both files"
    )
    args = parser.parse_args()
    from_path = args.__getattr__("from")
    move_database(
        from_path,
        args.to,
        exist_ok=args.exist_ok,
        overwrite=args.overwrite,
    )
