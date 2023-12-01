desc = "Build database from nuScenes/Lyft scenarios"

prediction_split = ["mini_train", "mini_val", "train", "train_val", "val"]
scene_split = ["v1.0-mini", "v1.0-trainval", "v1.0-test"]

if __name__ == "__main__":
    import pkg_resources  # for suppress warning
    import argparse
    import os.path
    from functools import partial
    from scenarionet import SCENARIONET_DATASET_PATH
    from scenarionet.converter.nuscenes.utils import convert_nuscenes_scenario, get_nuscenes_scenarios, \
        get_nuscenes_prediction_split
    from scenarionet.converter.utils import write_to_directory

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--database_path",
        "-d",
        default=os.path.join(SCENARIONET_DATASET_PATH, "nuscenes"),
        help="directory, The path to place the data"
    )
    parser.add_argument(
        "--dataset_name", "-n", default="nuscenes", help="Dataset name, will be used to generate scenario files"
    )
    parser.add_argument(
        "--split",
        default="v1.0-mini",
        choices=scene_split + prediction_split,
        help="Which splits of nuScenes data should be sued. If set to {}, it will convert the full log into scenarios"
        " with 20 second episode length. If set to {}, it will convert segments used for nuScenes prediction"
        " challenge to scenarios, resulting in more converted scenarios. Generally, you should choose this "
        " parameter from {} to get complete scenarios for planning unless you want to use the converted scenario "
        " files for prediction task.".format(scene_split, prediction_split, scene_split)
    )
    parser.add_argument("--dataroot", default="/data/sets/nuscenes", help="The path of nuscenes data")
    parser.add_argument("--map_radius", default=500, type=float, help="The size of map")
    parser.add_argument(
        "--future",
        default=6,
        help="6 seconds by default. How many future seconds to predict. Only "
        "available if split is chosen from {}".format(prediction_split)
    )
    parser.add_argument(
        "--past",
        default=2,
        help="2 seconds by default. How many past seconds are used for prediction."
        " Only available if split is chosen from {}".format(prediction_split)
    )
    parser.add_argument("--overwrite", action="store_true", help="If the database_path exists, whether to overwrite it")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers to use")
    args = parser.parse_args()

    overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.database_path
    version = args.split

    if version in scene_split:
        scenarios, nuscs = get_nuscenes_scenarios(args.dataroot, version, args.num_workers)
    else:
        scenarios, nuscs = get_nuscenes_prediction_split(
            args.dataroot, version, args.past, args.future, args.num_workers
        )
    write_to_directory(
        convert_func=convert_nuscenes_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        overwrite=overwrite,
        num_workers=args.num_workers,
        nuscenes=nuscs,
        past=[args.past for _ in range(args.num_workers)],
        future=[args.future for _ in range(args.num_workers)],
        prediction=[version in prediction_split for _ in range(args.num_workers)],
        map_radius=[args.map_radius for _ in range(args.num_workers)],
    )
