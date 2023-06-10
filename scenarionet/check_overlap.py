"""
Check If any overlap between two database
"""

import argparse

from scenarionet.common_utils import read_dataset_summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_1', type=str, required=True, help="The path of the first database")
    parser.add_argument('--database_2', type=str, required=True, help="The path of the second database")
    args = parser.parse_args()

    summary_1, _, _ = read_dataset_summary(args.database_1)
    summary_2, _, _ = read_dataset_summary(args.database_2)

    intersection = set(summary_1.keys()).intersection(set(summary_2.keys()))
    if len(intersection) == 0:
        print("No overlapping in two database!")
    else:
        print("Find overlapped scenarios: {}".format(intersection))
