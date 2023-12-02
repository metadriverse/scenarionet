desc = "Merge a list of databases. e.g. scenario.merge --from db_1 db_2 db_3...db_n --to db_dest"

if __name__ == '__main__':
    import argparse

    from scenarionet.builder.filters import ScenarioFilter
    from scenarionet.builder.utils import merge_database

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--database_path",
        "-d",
        "--to",
        required=True,
        help="The name of the new combined database. "
        "It will create a new directory to store dataset_summary.pkl and dataset_mapping.pkl. "
        "If exists_ok=True, those two .pkl files will be stored in an existing directory and turn "
        "that directory into a database."
    )
    parser.add_argument(
        '--from',
        required=True,
        nargs='+',
        default=[],
        help="Which datasets to combine. It takes any number of directory path as input"
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="Still allow to write, if the dir exists already. "
        "This write will only create two .pkl files and this directory will become a database."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="When exists ok is set but summary.pkl and map.pkl exists in existing dir, "
        "whether to overwrite both files"
    )
    parser.add_argument(
        "--filter_moving_dist",
        action="store_true",
        help="add this flag to select cases with SDC moving dist > sdc_moving_dist_min"
    )
    parser.add_argument(
        "--sdc_moving_dist_min",
        default=5,
        type=float,
        help="Selecting case with sdc_moving_dist > this value. "
        "We will add more filter conditions in the future."
    )
    args = parser.parse_args()
    target = args.sdc_moving_dist_min
    filters = [ScenarioFilter.make(ScenarioFilter.sdc_moving_dist, target_dist=target, condition="greater")]
    source = args.__getattribute__("from")
    if len(source) != 0:
        merge_database(
            args.database_path,
            *source,
            exist_ok=args.exist_ok,
            overwrite=args.overwrite,
            try_generate_missing_file=True,
            filters=filters if args.filter_moving_dist else []
        )
    else:
        raise ValueError("No source database are provided. Abort.")
