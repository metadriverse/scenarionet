desc = "Build database from Waymo scenarios"

if __name__ == '__main__':
    import shutil
    import argparse
    import logging
    import os
    import tensorflow as tf

    from scenarionet import SCENARIONET_DATASET_PATH, SCENARIONET_REPO_PATH
    from scenarionet.converter.utils import write_to_directory
    from scenarionet.converter.waymo.utils import convert_waymo_scenario, get_waymo_scenarios, \
        preprocess_waymo_scenarios

    tf.config.experimental.set_visible_devices([], "GPU")

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--database_path",
        "-d",
        default=os.path.join(SCENARIONET_DATASET_PATH, "waymo"),
        help="A directory, the path to place the converted data"
    )
    parser.add_argument(
        "--dataset_name", "-n", default="waymo", help="Dataset name, will be used to generate scenario files"
    )
    parser.add_argument("--version", "-v", default='v1.2', help="version")
    parser.add_argument("--overwrite", action="store_true", help="If the database_path exists, whether to overwrite it")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers to use")
    parser.add_argument(
        "--raw_data_path",
        default=os.path.join(SCENARIONET_REPO_PATH, "waymo_origin"),
        help="The directory stores all waymo tfrecord"
    )
    parser.add_argument(
        "--start_file_index",
        default=0,
        type=int,
        help="Control how many files to use. We will list all files in the raw data folder "
        "and select files[start_file_index: start_file_index+num_files]. Default: 0."
    )
    parser.add_argument(
        "--num_files",
        default=None,
        type=int,
        help="Control how many files to use. We will list all files in the raw data folder "
        "and select files[start_file_index: start_file_index+num_files]. Default: None, will read all files."
    )
    args = parser.parse_args()

    overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.database_path
    version = args.version

    save_path = output_path
    if os.path.exists(output_path):
        if not overwrite:
            raise ValueError(
                "Directory {} already exists! Abort. "
                "\n Try setting overwrite=True or adding --overwrite".format(output_path)
            )
        else:
            shutil.rmtree(output_path)

    waymo_data_directory = os.path.join(SCENARIONET_DATASET_PATH, args.raw_data_path)
    files = get_waymo_scenarios(waymo_data_directory, args.start_file_index, args.num_files)

    logger.info(
        f"We will read {len(files)} raw files. You set the number of workers to {args.num_workers}. "
        f"Please make sure there will not be too much files to be read in each worker "
        f"(now it's {len(files) / args.num_workers})!"
    )

    write_to_directory(
        convert_func=convert_waymo_scenario,
        scenarios=files,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        overwrite=overwrite,
        num_workers=args.num_workers,
        preprocess=preprocess_waymo_scenarios,
    )
