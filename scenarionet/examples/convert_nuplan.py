"""
This script aims to convert nuplan scenarios to ScenarioDescription, so that we can load any nuplan scenarios into
MetaDrive.
"""
from scenarionet import SCENARIONET_DATASET_PATH
import os
from scenarionet.converter.nuplan.utils import get_nuplan_scenarios, convert_nuplan_scenario
from scenarionet.converter.utils import write_to_directory

if __name__ == "__main__":
    # 14 types
    all_scenario_types = "[behind_pedestrian_on_pickup_dropoff,  \
                            near_multiple_vehicles, \
                            high_magnitude_jerk, \
                            crossed_by_vehicle, \
                            following_lane_with_lead, \
                            changing_lane_to_left, \
                            accelerating_at_traffic_light_without_lead, \
                            stopping_at_stop_sign_with_lead, \
                            traversing_narrow_lane, \
                            waiting_for_pedestrian_to_cross, \
                            starting_left_turn, \
                            starting_high_speed_turn, \
                            starting_unprotected_cross_turn, \
                            starting_protected_noncross_turn, \
                            on_pickup_dropoff]"

    dataset_params = [
        # builder setting
        "scenario_builder=nuplan_mini",
        "scenario_builder.scenario_mapping.subsample_ratio_override=0.5",  # 10 hz

        # filter
        "scenario_filter=all_scenarios",  # simulate only one log
        "scenario_filter.remove_invalid_goals=true",
        "scenario_filter.shuffle=true",
        "scenario_filter.log_names=['2021.07.16.20.45.29_veh-35_01095_01486']",
        # "scenario_filter.scenario_types={}".format(all_scenario_types),
        # "scenario_filter.scenario_tokens=[]",
        # "scenario_filter.map_names=[]",
        # "scenario_filter.num_scenarios_per_type=1",
        # "scenario_filter.limit_total_scenarios=1000",
        # "scenario_filter.expand_scenarios=true",
        # "scenario_filter.limit_scenarios_per_type=10",  # use 10 scenarios per scenario type
        "scenario_filter.timestamp_threshold_s=20",  # minial scenario duration (s)
    ]
    force_overwrite = True
    output_path = os.path.join(SCENARIONET_DATASET_PATH, "nuplan")
    version = 'v1.2'

    scenarios = get_nuplan_scenarios(dataset_params)
    write_to_directory(convert_func=convert_nuplan_scenario,
                       scenarios=scenarios,
                       output_path=output_path,
                       dataset_version=version,
                       dataset_name="nuscenes",
                       force_overwrite=force_overwrite,
                       )
