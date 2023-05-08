import argparse

from scenarionet.verifier.error import ErrorFile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", required=True, help="The path of the error file")
    parser.add_argument("--dataset_path", "-d", required=True, help="The path of the generated dataset")
    parser.add_argument("--overwrite", action="store_true", help="If the dataset_path exists, overwrite it")
    parser.add_argument("--broken", action="store_true", help="Generate dataset containing only broken files")
    args = parser.parse_args()
    ErrorFile.generate_dataset(args.file, args.dataset_path, args.overwrite, args.broken)
