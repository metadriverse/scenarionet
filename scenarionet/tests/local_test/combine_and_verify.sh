#!/usr/bin/env bash

python ../../scripts/combine_dataset.py --to_dataset ../../dataset/test_combine_dataset --from_dataset ../../dataset/waymo ../../dataset/pg ../../dataset/nuscenes ../../dataset/nuplan --overwrite
python ../../scripts/combine_dataset.py --dataset_path ../../dataset/test_combine_dataset --result_save_dir ../../dataset/test_combine_dataset