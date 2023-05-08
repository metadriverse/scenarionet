import os.path

import metadrive

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.pg.utils import get_pg_scenarios, convert_pg_scenario
from scenarionet.converter.utils import write_to_directory
from metadrive.policy.idm_policy import IDMPolicy
# from metadrive.policy.expert_policy import ExpertPolicy

if __name__ == '__main__':
    dataset_name = "pg"
    output_path = os.path.join(SCENARIONET_DATASET_PATH, dataset_name)
    version = metadrive.constants.DATA_VERSION
    force_overwrite = True

    scenario_indices, env = get_pg_scenarios(30, IDMPolicy)

    write_to_directory(
        convert_func=convert_pg_scenario,
        scenarios=scenario_indices,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        force_overwrite=force_overwrite,
        env=env
    )
