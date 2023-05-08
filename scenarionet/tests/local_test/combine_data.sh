#!/usr/bin/env bash

nohup python ../../scripts/combine_dataset.py --to_dataset ../../dataset/test_combine_dataset --from_dataset ../../dataset/waymo ../../dataset/pg ../../dataset/nuscenes ../../dataset/nuplan --overwrite > nuplan.log 2>&1 &