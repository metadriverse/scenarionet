"""
Check If any overlap between two database
"""
desc = "Check if there are overlapped scenarios between two databases. " \
       "If so, return the number of overlapped scenarios and id list"

if __name__ == '__main__':

    import argparse

    from scenarionet.common_utils import read_dataset_summary

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--d_1', type=str, required=True, help="The path of the first database")
    parser.add_argument('--d_2', type=str, required=True, help="The path of the second database")
    parser.add_argument('--show_id', action="store_true", help="whether to show the id of overlapped scenarios")
    args = parser.parse_args()

    summary_1, _, _ = read_dataset_summary(args.database_1)
    summary_2, _, _ = read_dataset_summary(args.database_2)

    intersection = set(summary_1.keys()).intersection(set(summary_2.keys()))
    if len(intersection) == 0:
        print("No overlapping in two database!")
    else:
        print("Find {} overlapped scenarios".format(len(intersection)))
        if args.show_id:
            print("Overlapped scenario ids:")
            for id in intersection:
                print(" " * 5, id)
