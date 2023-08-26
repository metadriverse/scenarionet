desc = "Check if the database is intact and all scenarios can be found and recorded in internal scenario description"

if __name__ == '__main__':
    import argparse

    from scenarionet.verifier.utils import verify_database, set_random_drop

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--database_path", "-d", required=True, help="Dataset path, a directory containing summary.pkl and mapping.pkl"
    )
    parser.add_argument(
        "--error_file_path",
        default="./",
        help="Where to save the error file. "
        "One can generate a new database excluding "
        "or only including the failed scenarios."
        "For more details, "
        "see operation 'generate_from_error_file'"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If an error file already exists in error_file_path, "
        "whether to overwrite it"
    )
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers to use")
    parser.add_argument("--random_drop", action="store_true", help="Randomly make some scenarios fail. for test only!")
    args = parser.parse_args()
    set_random_drop(args.random_drop)
    verify_database(
        args.database_path,
        args.error_file_path,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        steps_to_run=0
    )
