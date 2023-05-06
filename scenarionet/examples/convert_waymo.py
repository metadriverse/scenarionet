import logging
import os

from scenarionet.converter.utils import write_to_directory
from scenarionet.converter.waymo.utils import convert_waymo_scenario, get_waymo_scenarios

logger = logging.getLogger(__name__)

from scenarionet import SCENARIONET_DATASET_PATH

if __name__ == '__main__':
    force_overwrite = True
    dataset_name = "waymo"
    output_path = os.path.join(SCENARIONET_DATASET_PATH, dataset_name)
    version = 'v1.2'

    waymo_data_direction = os.path.join(SCENARIONET_DATASET_PATH, "waymo_origin")
    scenarios = get_waymo_scenarios(waymo_data_direction)

    write_to_directory(convert_func=convert_waymo_scenario,
                       scenarios=scenarios,
                       output_path=output_path,
                       dataset_version=version,
                       dataset_name=dataset_name,
                       force_overwrite=force_overwrite)
