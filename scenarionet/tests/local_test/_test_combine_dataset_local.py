import os

from scenarionet import SCENARIONET_DATASET_PATH
from scenarionet.builder.utils import combine_multiple_dataset
from scenarionet.verifier.utils import verify_loading_into_metadrive


def _test_combine_dataset():
    dataset_paths = [
        os.path.join(SCENARIONET_DATASET_PATH, "nuscenes"),
        os.path.join(SCENARIONET_DATASET_PATH, "nuplan"),
        os.path.join(SCENARIONET_DATASET_PATH, "waymo"),
        os.path.join(SCENARIONET_DATASET_PATH, "pg")
    ]

    combine_path = os.path.join(SCENARIONET_DATASET_PATH, "combined_dataset")
    combine_multiple_dataset(combine_path, *dataset_paths, force_overwrite=True, try_generate_missing_file=True)
    # os.makedirs("verify_results", exist_ok=True)
    # verify_loading_into_metadrive(combine_path, "verify_results")
    # assert success


if __name__ == '__main__':
    _test_combine_dataset()
