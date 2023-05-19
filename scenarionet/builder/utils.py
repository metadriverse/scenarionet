import copy
import logging
import os
import os.path as osp
import pickle
import shutil
from typing import Callable, List

import tqdm
from metadrive.scenario.scenario_description import ScenarioDescription

from scenarionet.common_utils import save_summary_anda_mapping

logger = logging.getLogger(__name__)


def try_generating_summary(file_folder):
    # Create a fake one
    files = os.listdir(file_folder)
    summary = {}
    for file in files:
        if ScenarioDescription.is_scenario_file(file):
            with open(osp.join(file_folder, file), "rb+") as f:
                scenario = pickle.load(f)
            summary[file] = copy.deepcopy(scenario[ScenarioDescription.METADATA])
    return summary


def merge_database(
    output_path,
    *dataset_paths,
    exist_ok=False,
    overwrite=False,
    try_generate_missing_file=True,
    filters: List[Callable] = None
):
    """
    Combine multiple datasets. Each dataset should have a dataset_summary.pkl
    :param output_path: The path to store the output dataset
    :param exist_ok: If True, though the output_path already exist, still write into it
    :param overwrite: If True, overwrite existing dataset_summary.pkl and mapping.pkl. Otherwise, raise error
    :param try_generate_missing_file: If dataset_summary.pkl and mapping.pkl are missing, whether to try generating them
    :param dataset_paths: Path of each dataset
    :param filters: a set of filters to choose which scenario to be selected and added into this combined dataset
    :return: summary, mapping
    """
    filters = filters or []
    output_abs_path = osp.abspath(output_path)
    os.makedirs(output_abs_path, exist_ok=exist_ok)
    summary_file = osp.join(output_abs_path, ScenarioDescription.DATASET.SUMMARY_FILE)
    mapping_file = osp.join(output_abs_path, ScenarioDescription.DATASET.MAPPING_FILE)
    for file in [summary_file, mapping_file]:
        if os.path.exists(file):
            if overwrite:
                os.remove(file)
            else:
                raise FileExistsError("{} already exists at: {}!".format(file, output_abs_path))

    summaries = {}
    mappings = {}

    # collect
    for dataset_path in tqdm.tqdm(dataset_paths):
        abs_dir_path = osp.abspath(dataset_path)
        # summary
        assert osp.exists(abs_dir_path), "Wrong dataset path. Can not find dataset at: {}".format(abs_dir_path)
        if not osp.exists(osp.join(abs_dir_path, ScenarioDescription.DATASET.SUMMARY_FILE)):
            if try_generate_missing_file:
                summary = try_generating_summary(abs_dir_path)
            else:
                raise FileNotFoundError("Can not find summary file for dataset: {}".format(abs_dir_path))
        else:
            with open(osp.join(abs_dir_path, ScenarioDescription.DATASET.SUMMARY_FILE), "rb+") as f:
                summary = pickle.load(f)
        intersect = set(summaries.keys()).intersection(set(summary.keys()))
        if len(intersect) > 0:
            existing = []
            for v in list(intersect):
                existing.append(mappings[v])
            logging.warning("Repeat scenarios: {} in : {}. Existing: {}".format(intersect, abs_dir_path, existing))
        summaries.update(summary)

        # mapping
        if not osp.exists(osp.join(abs_dir_path, ScenarioDescription.DATASET.MAPPING_FILE)):
            if try_generate_missing_file:
                mapping = {k: "" for k in summary}
            else:
                raise FileNotFoundError("Can not find mapping file for dataset: {}".format(abs_dir_path))
        else:
            with open(osp.join(abs_dir_path, ScenarioDescription.DATASET.MAPPING_FILE), "rb+") as f:
                mapping = pickle.load(f)
        new_mapping = {}
        for file, rel_path in mapping.items():
            # mapping to real file path
            new_mapping[file] = os.path.relpath(osp.join(abs_dir_path, rel_path), output_abs_path)

        mappings.update(new_mapping)

    # apply filter stage
    file_to_pop = []
    for file_name, metadata, in summaries.items():
        if not all([fil(metadata) for fil in filters]):
            file_to_pop.append(file_name)
    for file in file_to_pop:
        summaries.pop(file)
        mappings.pop(file)

    save_summary_anda_mapping(summary_file, mapping_file, summaries, mappings)

    return summaries, mappings


def move_database(
    from_path,
    to_path,
    exist_ok=False,
    overwrite=False,
):
    if not os.path.exists(from_path):
        raise FileNotFoundError("Can not find dataset: {}".format(from_path))
    if os.path.exists(to_path):
        assert exist_ok, "to_directory already exists. Set exists_ok to allow turning it into a dataset"
        assert not os.path.samefile(from_path, to_path), "to_directory is the same as from_directory. Abort!"
    merge_database(
        to_path,
        from_path,
        exist_ok=exist_ok,
        overwrite=overwrite,
        try_generate_missing_file=True,
    )
    files = os.listdir(from_path)
    if ScenarioDescription.DATASET.MAPPING_FILE in files and ScenarioDescription.DATASET.SUMMARY_FILE in files and len(
            files) == 2:
        shutil.rmtree(from_path)
