#!/usr/bin/env bash

python ../../merge.py --overwrite --exist_ok --database_path ../tmp/test_combine_dataset --from ../../../dataset/waymo ../../../dataset/pg ../../../dataset/nuscenes ../../../dataset/nuplan --overwrite
python ../../check_simulation.py --overwrite --database_path ../tmp/test_combine_dataset --error_file_path ../tmp/test_combine_dataset --random_drop --num_workers=16
python ../../generate_from_error_file.py --file ../tmp/test_combine_dataset/error_scenarios_for_test_combine_dataset.json --overwrite --database_path ../tmp/verify_pass
python ../../generate_from_error_file.py --file ../tmp/test_combine_dataset/error_scenarios_for_test_combine_dataset.json --overwrite --database_path ../tmp/verify_fail --broken