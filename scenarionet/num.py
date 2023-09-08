desc = "The number of scenarios in the specified database"

if __name__ == '__main__':
    import pkg_resources  # for suppress warning
    import argparse
    import logging
    from scenarionet.common_utils import read_dataset_summary

    logger = logging.getLogger(__file__)

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--database_path", "-d", required=True, help="Database to check number of scenarios")
    args = parser.parse_args()
    summary, _, _, = read_dataset_summary(args.database_path)
    logger.info("Number of scenarios: {}".format(len(summary)))
