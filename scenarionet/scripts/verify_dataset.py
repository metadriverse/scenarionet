import pkg_resources  # for suppress warning
import argparse
from scenarionet.common_utils import read_dataset_summary, read_scenario
from metadrive.scenario.scenario_description import ScenarioDescription as SD
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", required=True, help="Dataset path, a directory containing summary.pkl and mapping.pkl"
    )
    args = parser.parse_args()
    summary, _, mapping = read_dataset_summary(args.dataset_path)
    for file_name in tqdm.tqdm(summary.keys()):
        try:
            scenario = read_scenario(args.dataset_path, mapping, file_name)
            SD.sanity_check(scenario)
        except Exception as e:
            raise ValueError("The file {} is broken, due to {}".format(file_name, str(e)))

            file_path = os.path.join(dataset_path, env.engine.data_manager.mapping[file_name], file_name)
            error_msg = ED.make(scenario_index, file_path, file_name, str(e))
            error_msgs.append(error_msg)
            success = False