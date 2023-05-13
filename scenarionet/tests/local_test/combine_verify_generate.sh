#!/usr/bin/env bash

python ../../combine_dataset.py --overwrite --exist_ok --dataset_path ../tmp/test_combine_dataset --from_datasets ../../../dataset/waymo ../../../dataset/pg ../../../dataset/nuscenes ../../../dataset/nuplan --overwrite
python ../../verify_simulation.py --overwrite --dataset_path ../tmp/test_combine_dataset --result_save_dir ../tmp/test_combine_dataset --random_drop --num_workers=16
python ../../generate_from_error_file.py --file ../tmp/test_combine_dataset/error_scenarios_for_test_combine_dataset.json --overwrite --dataset_path ../tmp/verify_pass
python ../../generate_from_error_file.py --file ../tmp/test_combine_dataset/error_scenarios_for_test_combine_dataset.json --overwrite --dataset_path ../tmp/verify_fail --broken