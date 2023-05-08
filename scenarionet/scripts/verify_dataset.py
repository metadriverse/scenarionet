"""
This script aims to convert nuscenes scenarios to ScenarioDescription, so that we can load any nuscenes scenarios into
MetaDrive.
"""
import argparse

from scenarionet.verifier.utils import verify_loading_into_metadrive

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="Dataset path, a directory")
    parser.add_argument("--result_save_dir", required=True, help="Dataset path, a directory")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers to use")
    args = parser.parse_args()
    verify_loading_into_metadrive(args.dataset_path, args.result_save_dir,         num_workers=args.num_workers)
