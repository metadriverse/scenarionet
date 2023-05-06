"""
This script takes --folder as input. It is the folder storing a batch of tfrecord file.
This script will create the output folder "processed_data" sharing the same level as `--folder`.

-- folder
-- processed_data

"""
import argparse
import copy
import os
import pickle

import numpy as np

from scenarionet.converter.utils import dict_recursive_remove_array, get_agent_summary, get_number_summary

try:
    import tensorflow as tf
except ImportError:
    pass

try:
    from waymo_open_dataset.protos import scenario_pb2
except ImportError:
    try:
        from metadrive.utils.waymo.protos import scenario_pb2  # Local files that only in PZH's computer.
    except ImportError:
        print(
            "Please install waymo_open_dataset package through metadrive dependencies: "
            "pip install waymo-open-dataset-tf-2-11-0==1.5.0"
        )

from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from scenarionet.converter.waymo.utils import extract_tracks, extract_dynamic_map_states, extract_map_features, \
    compute_width
import sys


def convert_waymo(file_list, input_path, output_path, worker_index=None):
    scenario = scenario_pb2.Scenario()

    metadata_recorder = {}

    total_scenarios = 0

    desc = ""
    summary_file = "dataset_summary.pkl"
    if worker_index is not None:
        desc += "Worker {} ".format(worker_index)
        summary_file = "dataset_summary_worker{}.pkl".format(worker_index)

    for file_count, file in enumerate(file_list):
        file_path = os.path.join(input_path, file)
        if ("tfrecord" not in file_path) or (not os.path.isfile(file_path)):
            continue
        dataset = tf.data.TFRecordDataset(file_path, compression_type="")

        total = sum(1 for _ in dataset.as_numpy_iterator())

        for j, data in enumerate(dataset.as_numpy_iterator()):
            scenario.ParseFromString(data)

            md_scenario = SD()

            md_scenario[SD.ID] = scenario.scenario_id
            # TODO LQY, get version from original files
            md_scenario[SD.VERSION] = "1.2"

            # Please note that SDC track index is not identical to sdc_id.
            # sdc_id is a unique indicator to a track, while sdc_track_index is only the index of the sdc track
            # in the tracks datastructure.

            track_length = len(list(scenario.timestamps_seconds))

            tracks, sdc_id = extract_tracks(scenario.tracks, scenario.sdc_track_index, track_length)

            md_scenario[SD.LENGTH] = track_length

            md_scenario[SD.TRACKS] = tracks

            dynamic_states = extract_dynamic_map_states(scenario.dynamic_map_states, track_length)

            md_scenario[SD.DYNAMIC_MAP_STATES] = dynamic_states

            map_features = extract_map_features(scenario.map_features)
            md_scenario[SD.MAP_FEATURES] = map_features

            compute_width(md_scenario[SD.MAP_FEATURES])

            md_scenario[SD.METADATA] = {}
            md_scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
            md_scenario[SD.METADATA][SD.TIMESTEP] = np.asarray(list(scenario.timestamps_seconds), dtype=np.float32)
            md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
            md_scenario[SD.METADATA][SD.SDC_ID] = str(sdc_id)
            md_scenario[SD.METADATA]["dataset"] = "waymo"
            md_scenario[SD.METADATA]["scenario_id"] = scenario.scenario_id
            md_scenario[SD.METADATA]["source_file"] = str(file)
            md_scenario[SD.METADATA]["track_length"] = track_length

            # === Waymo specific data. Storing them here ===
            md_scenario[SD.METADATA]["current_time_index"] = scenario.current_time_index
            md_scenario[SD.METADATA]["sdc_track_index"] = scenario.sdc_track_index

            # obj id
            md_scenario[SD.METADATA]["objects_of_interest"] = [str(obj) for obj in scenario.objects_of_interest]

            track_index = [obj.track_index for obj in scenario.tracks_to_predict]
            track_id = [str(scenario.tracks[ind].id) for ind in track_index]
            track_difficulty = [obj.difficulty for obj in scenario.tracks_to_predict]
            track_obj_type = [tracks[id]["type"] for id in track_id]
            md_scenario[SD.METADATA]["tracks_to_predict"] = {
                id: {
                    "track_index": track_index[count],
                    "track_id": id,
                    "difficulty": track_difficulty[count],
                    "object_type": track_obj_type[count]
                }
                for count, id in enumerate(track_id)
            }

            export_file_name = "sd_{}_{}.pkl".format(file, scenario.scenario_id)

            summary_dict = {}
            summary_dict["sdc"] = get_agent_summary(
                state_dict=md_scenario.get_sdc_track()["state"], id=sdc_id, type=md_scenario.get_sdc_track()["type"]
            )
            for track_id, track in md_scenario[SD.TRACKS].items():
                summary_dict[track_id] = get_agent_summary(state_dict=track["state"], id=track_id, type=track["type"])
            md_scenario[SD.METADATA]["object_summary"] = summary_dict

            # Count some objects occurrence
            md_scenario[SD.METADATA]["number_summary"] = get_number_summary(md_scenario)

            metadata_recorder[export_file_name] = copy.deepcopy(md_scenario[SD.METADATA])

            md_scenario = md_scenario.to_dict()

            SD.sanity_check(md_scenario, check_self_type=True)

            p = os.path.join(output_path, export_file_name)
            with open(p, "wb") as f:
                pickle.dump(md_scenario, f)

            total_scenarios += 1
            if j == total - 1:
                print(
                    f"{desc}Collected {total_scenarios} scenarios. File {file_count + 1}/{len(file_list)} has "
                    f"{total} Scenarios. The last one is saved at: {p}"
                )

    summary_file = os.path.join(output_path, summary_file)
    with open(summary_file, "wb") as file:
        pickle.dump(dict_recursive_remove_array(metadata_recorder), file)
    print("Summary is saved at: {}".format(summary_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="The data folder storing raw tfrecord from Waymo dataset.")
    parser.add_argument(
        "--output", default="processed_data", type=str, help="The data folder storing raw tfrecord from Waymo dataset."
    )
    args = parser.parse_args()

    scenario_data_path = args.input

    output_path: str = os.path.dirname(scenario_data_path)
    output_path = os.path.join(output_path, args.output)
    os.makedirs(output_path, exist_ok=True)

    raw_data_path = scenario_data_path

    # parse raw data from input path to output path,
    # there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    file_list = os.listdir(raw_data_path)
    convert_waymo(file_list, raw_data_path, output_path)
    sys.exit()
    # file_path = AssetLoader.file_path("waymo", "processed", "0.pkl", return_raw_style=False)
    # data = read_waymo_data(file_path)
    # draw_waymo_map(data)
