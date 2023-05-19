import pkg_resources  # for suppress warning
import argparse
from scenarionet.verifier.utils import verify_database, set_random_drop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database_path", "-d", required=True, help="Dataset path, a directory containing summary.pkl and mapping.pkl"
    )
    parser.add_argument("--result_save_dir", default="./", help="Where to save the error file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If an error file already exists in result_save_dir, "
        "whether to overwrite it"
    )
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers to use")
    parser.add_argument("--random_drop", action="store_true", help="Randomly make some scenarios fail. for test only!")
    args = parser.parse_args()
    set_random_drop(args.random_drop)
    verify_database(args.database_path, args.result_save_dir, overwrite=args.overwrite, num_workers=args.num_workers)
