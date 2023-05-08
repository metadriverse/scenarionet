import argparse

from scenarionet.verifier.utils import verify_loading_into_metadrive, set_random_drop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="Dataset path, a directory")
    parser.add_argument("--result_save_dir", required=True, help="Dataset path, a directory")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers to use")
    parser.add_argument("--random_drop", action="store_true", help="Randomly make some scenarios fail. for test only!")
    args = parser.parse_args()
    set_random_drop(args.random_drop)
    verify_loading_into_metadrive(args.dataset_path, args.result_save_dir, num_workers=args.num_workers)
