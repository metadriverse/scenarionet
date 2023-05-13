import pkg_resources  # for suppress warning
import argparse

from scenarionet.verifier.error import ErrorFile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", required=True, help="The path of the error file, should be xyz.json")
    parser.add_argument("--dataset_path", "-d", required=True, help="The path of the newly generated dataset")
    parser.add_argument("--overwrite", action="store_true", help="If the dataset_path exists, overwrite it")
    parser.add_argument(
        "--broken",
        action="store_true",
        help="By default, only successful scenarios will be picked to build the new dataset. "
        "If turn on this flog, it will generate dataset containing only broken scenarios."
    )
    args = parser.parse_args()
    ErrorFile.generate_dataset(args.file, args.dataset_path, args.overwrite, args.broken)
