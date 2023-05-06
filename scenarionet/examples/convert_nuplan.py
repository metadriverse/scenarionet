"""
This script aims to convert nuplan scenarios to ScenarioDescription, so that we can load any nuplan scenarios into
MetaDrive.
"""
import os

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.nuplan.utils import get_nuplan_scenarios, convert_nuplan_scenario, example_dataset_params
from scenarionet.converter.utils import write_to_directory

if __name__ == "__main__":
    force_overwrite = True
    dataset_name = "nuplan"
    output_path = os.path.join(SCENARIONET_DATASET_PATH, dataset_name)
    version = 'v1.1'

    scenarios = get_nuplan_scenarios(example_dataset_params)

    write_to_directory(convert_func=convert_nuplan_scenario,
                       scenarios=scenarios,
                       output_path=output_path,
                       dataset_version=version,
                       dataset_name=dataset_name,
                       force_overwrite=force_overwrite,
                       )
