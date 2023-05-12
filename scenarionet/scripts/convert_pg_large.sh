#!/bin/bash

# check if four arguments are passed
if [ $# -ne 4 ]; then
    echo "Usage: $0 <num_scenarios_each_sub_dataset> <num_sub_datasets> <dataset_path> <num_workers>"
    exit 1
fi

# get the number of scenarios, datasets, dataset path, and number of workers from command line arguments
num_scenarios_each_sub_dataset=$1
num_sub_datasets=$2
dataset_path=$3
num_workers=$4

# run the conversion script in a loop
for i in $(seq 0 $((num_sub_datasets-1)))
do
  start_index=$(($num_scenarios * $i))
  sub_dataset_path="${dataset_path}/pg_$i"
  python -m scenarionet.scripts.convert_pg -n pg -d $sub_dataset_path --start_index=$start_index --num_workers=$num_workers --num_scenarios_each_sub_dataset=$num_scenarios
done

# combine the datasets
python -m scenarionet.scripts.combine_dataset dataset_path $dataset_path --from_datasets $(for i in $(seq 0 $((num_sub_datasets-1))); do echo -n "${dataset_path}/pg_$i "; done)
