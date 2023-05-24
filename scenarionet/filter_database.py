import pkg_resources  # for suppress warning
import argparse
from scenarionet.builder.filters import ScenarioFilter
from scenarionet.builder.utils import merge_database

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        '--from_dataset',
        required=True,
        type=str,
        help="Which dataset to filter. It takes one directory path as input"
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
        default=10,
        type=float,
        help="Selecting case with sdc_moving_dist > this value. "
             "We will add more filter conditions in the future."
    )

    parser.add_argument(
        "--filter_num_object",
        action="store_true",
        help="add this flag to select cases with object_num < max_num_object"
    )
    parser.add_argument(
        "--max_num_object",
        default=30,
        type=float,
        help="case will be selected if num_obj < this argument"
    )

    parser.add_argument(
        "--filter_overpass",
        action="store_true",
        help="Scenarios with overpass WON'T be selected"
    )

    args = parser.parse_args()
    target = args.sdc_moving_dist_min
    obj_threshold = args.max_num_object
    filters = [ScenarioFilter.make(ScenarioFilter.sdc_moving_dist, target_dist=target, condition="greater"),
               ScenarioFilter.make(ScenarioFilter.object_number, number_threshold=obj_threshold)]

    if len(args.from_datasets) != 0:
        merge_database(
            args.database_path,
            *args.from_datasets,
            exist_ok=args.exist_ok,
            overwrite=args.overwrite,
            try_generate_missing_file=True,
            filters=filters if args.filter_moving_dist else []
        )
    else:
        raise ValueError("No source database are provided. Abort.")
