# from metadrive.policy.expert_policy import ExpertPolicy
import argparse
import os.path

import metadrive
from metadrive.policy.idm_policy import IDMPolicy

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.converter.pg.utils import get_pg_scenarios, convert_pg_scenario
from scenarionet.converter.utils import write_to_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-n", default="pg",
                        help="Dataset name, will be used to generate scenario files")
    parser.add_argument("--dataset_path", "-d", default=os.path.join(SCENARIONET_DATASET_PATH, "pg"),
                        help="The path of the dataset")
    parser.add_argument("--version", "-v", default=metadrive.constants.DATA_VERSION,  help="version")
    parser.add_argument("--overwrite", action="store_true", help="If the dataset_path exists, overwrite it")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers to use")
    args = parser.parse_args()

    force_overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.dataset_path
    version = args.version

    scenario_indices, env = get_pg_scenarios(30, IDMPolicy)

    write_to_directory(
        convert_func=convert_pg_scenario,
        scenarios=scenario_indices,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        force_overwrite=force_overwrite,
        env=env,
        num_workers=args.num_workers
    )
