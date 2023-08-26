desc = "Move or Copy an existing database"

if __name__ == '__main__':
    import argparse

    from scenarionet.builder.utils import copy_database

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--from', required=True, help="Which database to move.")
    parser.add_argument(
        "--to",
        required=True,
        help="The name of the new database. "
        "It will create a new directory to store dataset_summary.pkl and dataset_mapping.pkl. "
        "If exists_ok=True, those two .pkl files will be stored in an existing directory and turn "
        "that directory into a database."
    )
    parser.add_argument("--remove_source", action="store_true", help="Remove the `from_database` if set this flag")
    parser.add_argument(
        "--copy_raw_data",
        action="store_true",
        help="Instead of creating virtual file mapping, copy raw scenario.pkl file"
    )
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
    from_path = args.__getattribute__("from")
    copy_database(
        from_path,
        args.to,
        exist_ok=args.exist_ok,
        overwrite=args.overwrite,
        copy_raw_data=args.copy_raw_data,
        remove_source=args.remove_source
    )
