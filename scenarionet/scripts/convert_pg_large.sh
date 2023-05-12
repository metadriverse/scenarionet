#!/bin/bash
# Author: GPT-4
# Usage: ./script_name.sh 5000 10 /path/to/datasets 8 true

# check if five arguments are passed
if [ $# -ne 5 ]; then
    echo "Usage: $0 <num_scenarios_sub_dataset> <num_sub_dataset> <dataset_path> <num_workers> <overwrite>"
    exit 1
fi

# get the number of scenarios, datasets, dataset path, number of workers, and overwrite from command line arguments
num_scenarios_sub_dataset=$1
num_sub_dataset=$2
dataset_path=$3
num_workers=$4
overwrite=$5

# initialize start_index
start_index=0

# run the conversion script in a loop
for i in $(seq 1 $num_sub_dataset)
do
  sub_dataset_path="${dataset_path}/pg_$((i-1))"
  if [ "$overwrite" = true ]; then
    python -m scenarionet.scripts.convert_pg -n pg -d $sub_dataset_path --start_index=$start_index --num_workers=$num_workers --num_scenarios=$num_scenarios_sub_dataset --overwrite
  else
    python -m scenarionet.scripts.convert_pg -n pg -d $sub_dataset_path --start_index=$start_index --num_workers=$num_workers --num_scenarios=$num_scenarios_sub_dataset
  fi
  start_index=$((start_index + num_scenarios_sub_dataset))
done

# combine the datasets
if [ "$overwrite" = true ]; then
  python -m scenarionet.scripts.combine_dataset --dataset_path $dataset_path --from_datasets $(for i in $(seq 0 $((num_sub_dataset-1))); do echo -n "${dataset_path}/pg_$i "; done) --overwrite --exist_ok
else
  python -m scenarionet.scripts.combine_dataset --dataset_path $dataset_path --from_datasets $(for i in $(seq 0 $((num_sub_dataset-1))); do echo -n "${dataset_path}/pg_$i "; done) --exist_ok
fi

