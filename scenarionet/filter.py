desc = "Filter unwanted scenarios out and build a new database"

if __name__ == '__main__':
    import argparse

    from scenarionet.builder.filters import ScenarioFilter
    from scenarionet.builder.utils import merge_database

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--database_path",
        "-d",
        required=True,
        help="The name of the new database. "
        "It will create a new directory to store dataset_summary.pkl and dataset_mapping.pkl. "
        "If exists_ok=True, those two .pkl files will be stored in an existing directory and turn "
        "that directory into a database."
    )
    parser.add_argument(
        '--from', required=True, type=str, help="Which dataset to filter. It takes one directory path as input"
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
        "--moving_dist",
        action="store_true",
        help="add this flag to select cases with SDC moving dist > sdc_moving_dist_min"
    )
    parser.add_argument(
        "--sdc_moving_dist_min", default=10, type=float, help="Selecting case with sdc_moving_dist > this value. "
    )

    parser.add_argument(
        "--num_object", action="store_true", help="add this flag to select cases with object_num < max_num_object"
    )
    parser.add_argument(
        "--max_num_object", default=30, type=float, help="case will be selected if num_obj < this argument"
    )

    parser.add_argument("--no_overpass", action="store_true", help="Scenarios with overpass WON'T be selected")

    parser.add_argument(
        "--no_traffic_light", action="store_true", help="Scenarios with traffic light WON'T be selected"
    )

    parser.add_argument("--id_filter", action="store_true", help="Scenarios with indicated name will NOT be selected")

    parser.add_argument(
        "--exclude_ids", nargs='+', default=[], help="Scenarios with indicated name will NOT be selected"
    )

    args = parser.parse_args()
    target = args.sdc_moving_dist_min
    obj_threshold = args.max_num_object
    from_path = args.__getattribute__("from")

    filters = []
    if args.no_overpass:
        filters.append(ScenarioFilter.make(ScenarioFilter.no_overpass))
    if args.num_object:
        filters.append(ScenarioFilter.make(ScenarioFilter.object_number, number_threshold=obj_threshold))
    if args.moving_dist:
        filters.append(ScenarioFilter.make(ScenarioFilter.sdc_moving_dist, target_dist=target, condition="greater"))
    if args.no_traffic_light:
        filters.append(ScenarioFilter.make(ScenarioFilter.no_traffic_light))
    if args.id_filter:
        filters.append(ScenarioFilter.make(ScenarioFilter.id_filter, ids=args.exclude_ids))

    if len(filters) == 0:
        raise ValueError("No filters are applied. Abort.")

    merge_database(
        args.database_path,
        from_path,
        exist_ok=args.exist_ok,
        overwrite=args.overwrite,
        try_generate_missing_file=True,
        filters=filters
    )
