desc = "Generate a new database excluding " \
       "or only including the failed scenarios detected by 'check_simulation' and 'check_existence'"

if __name__ == '__main__':
    import pkg_resources  # for suppress warning
    import argparse

    from scenarionet.verifier.error import ErrorFile

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--database_path", "-d", required=True, help="The path of the newly generated database")
    parser.add_argument("--file", "-f", required=True, help="The path of the error file, should be xyz.json")
    parser.add_argument("--overwrite", action="store_true", help="If the database_path exists, overwrite it")
    parser.add_argument(
        "--broken",
        action="store_true",
        help="By default, only successful scenarios will be picked to build the new database. "
        "If turn on this flog, it will generate database containing only broken scenarios."
    )
    args = parser.parse_args()
    ErrorFile.generate_dataset(args.file, args.database_path, args.overwrite, args.broken)
