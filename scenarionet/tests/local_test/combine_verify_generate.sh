#!/usr/bin/env bash

python ../../scripts/combine_dataset.py --to ../tmp/test_combine_dataset --from_datasets ../../../dataset/waymo ../../../dataset/pg ../../../dataset/nuscenes ../../../dataset/nuplan --overwrite
python ../../scripts/verify_dataset.py --dataset_path ../tmp/test_combine_dataset --result_save_dir ../tmp/test_combine_dataset
python ../../scripts/generate_from_error_file.py --file ../tmp/test_combine_dataset/errors_test_combine_dataset --overwrite --dataset_path ../tmp/verify_pass
python ../../scripts/generate_from_error_file.py --file ../tmp/test_combine_dataset/errors_test_combine_dataset --overwrite --dataset_path ../tmp/verify_fail --broken