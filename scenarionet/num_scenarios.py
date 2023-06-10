import pkg_resources  # for suppress warning
import argparse
import logging
from scenarionet.common_utils import read_dataset_summary

logger = logging.getLogger(__file__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database_path",
        "-d",
        required=True,
        help="Database to check number of scenarios"
    )
    args = parser.parse_args()
    summary, _, _, = read_dataset_summary(args.database_path)
    logger.info("Number of scenarios: {}".format(len(summary)))
