import copy
from random import sample
from metadrive.scenario.utils import read_dataset_summary
import logging
import os
import os.path as osp
import pickle
import shutil
from typing import Callable, List

import tqdm
from metadrive.scenario.scenario_description import ScenarioDescription

from scenarionet.common_utils import save_summary_and_mapping

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
    filters: List[Callable] = None,
    save=True,
):
    """
    Combine multiple datasets. Each database should have a dataset_summary.pkl
    :param output_path: The path to store the output database
    :param exist_ok: If True, though the output_path already exist, still write into it
    :param overwrite: If True, overwrite existing dataset_summary.pkl and mapping.pkl. Otherwise, raise error
    :param try_generate_missing_file: If dataset_summary.pkl and mapping.pkl are missing, whether to try generating them
    :param dataset_paths: Path of each database
    :param filters: a set of filters to choose which scenario to be selected and added into this combined database
    :param save: save to output path, immediately
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
    for dataset_path in tqdm.tqdm(dataset_paths, desc="Merge Data"):
        abs_dir_path = osp.abspath(dataset_path)
        # summary
        assert osp.exists(abs_dir_path), "Wrong database path. Can not find database at: {}".format(abs_dir_path)
        if not osp.exists(osp.join(abs_dir_path, ScenarioDescription.DATASET.SUMMARY_FILE)):
            if try_generate_missing_file:
                summary = try_generating_summary(abs_dir_path)
            else:
                raise FileNotFoundError("Can not find summary file for database: {}".format(abs_dir_path))
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
                raise FileNotFoundError("Can not find mapping file for database: {}".format(abs_dir_path))
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
    for file_name in tqdm.tqdm(summaries.keys(), desc="Filter Scenarios"):
        metadata = summaries[file_name]
        if not all([fil(metadata, os.path.join(output_abs_path, mappings[file_name], file_name)) for fil in filters]):
            file_to_pop.append(file_name)
    for file in file_to_pop:
        summaries.pop(file)
        mappings.pop(file)
    if save:
        save_summary_and_mapping(summary_file, mapping_file, summaries, mappings)

    return summaries, mappings


def copy_database(from_path, to_path, exist_ok=False, overwrite=False, copy_raw_data=False, remove_source=False):
    if not os.path.exists(from_path):
        raise FileNotFoundError("Can not find database: {}".format(from_path))
    if os.path.exists(to_path):
        assert exist_ok, "to_directory already exists. Set exists_ok to allow turning it into a database"
        assert not os.path.samefile(from_path, to_path), "to_directory is the same as from_directory. Abort!"
    files = os.listdir(from_path)
    official_file_num = sum(
        [ScenarioDescription.DATASET.MAPPING_FILE in files, ScenarioDescription.DATASET.SUMMARY_FILE in files]
    )
    if remove_source and len(files) > official_file_num:
        raise RuntimeError(
            "The source database is not allowed to move! "
            "This will break the relationship between this database and other database built on it."
            "If it is ok for you, use 'mv' to move it manually "
        )

    summaries, mappings = merge_database(
        to_path, from_path, exist_ok=exist_ok, overwrite=overwrite, try_generate_missing_file=True, save=False
    )
    summary_file = osp.join(to_path, ScenarioDescription.DATASET.SUMMARY_FILE)
    mapping_file = osp.join(to_path, ScenarioDescription.DATASET.MAPPING_FILE)

    if copy_raw_data:
        logger.info("Copy raw data...")
        for scenario_file in tqdm.tqdm(mappings.keys()):
            rel_path = mappings[scenario_file]
            shutil.copyfile(os.path.join(to_path, rel_path, scenario_file), os.path.join(to_path, scenario_file))
        mappings = {key: "./" for key in summaries.keys()}
    save_summary_and_mapping(summary_file, mapping_file, summaries, mappings)

    if remove_source:
        if ScenarioDescription.DATASET.MAPPING_FILE in files and ScenarioDescription.DATASET.SUMMARY_FILE in files \
                and len(files) == 2:
            shutil.rmtree(from_path)
            logger.info("Successfully remove: {}".format(from_path))
        else:
            logger.info(
                "Failed to remove: {}, as it might contain scenario files "
                "or has no summary file or mapping file".format(from_path)
            )


def split_database(
    from_path,
    to_path,
    start_index,
    num_scenarios,
    exist_ok=False,
    overwrite=False,
    random=False,
):
    if not os.path.exists(from_path):
        raise FileNotFoundError("Can not find database: {}".format(from_path))
    if os.path.exists(to_path):
        assert exist_ok, "to_directory already exists. Set exists_ok to allow turning it into a database"
        assert not os.path.samefile(from_path, to_path), "to_directory is the same as from_directory. Abort!"
    overwrite = overwrite,
    output_abs_path = osp.abspath(to_path)
    os.makedirs(output_abs_path, exist_ok=exist_ok)
    summary_file = osp.join(output_abs_path, ScenarioDescription.DATASET.SUMMARY_FILE)
    mapping_file = osp.join(output_abs_path, ScenarioDescription.DATASET.MAPPING_FILE)
    for file in [summary_file, mapping_file]:
        if os.path.exists(file):
            if overwrite:
                os.remove(file)
            else:
                raise FileExistsError("{} already exists at: {}!".format(file, output_abs_path))

    # collect
    abs_dir_path = osp.abspath(from_path)
    # summary
    assert osp.exists(abs_dir_path), "Wrong database path. Can not find database at: {}".format(abs_dir_path)
    summaries, lookup, mappings = read_dataset_summary(from_path)
    assert start_index >= 0 and start_index + num_scenarios <= len(
        lookup
    ), "No enough scenarios in source dataset: total {}, start_index: {}, need: {}".format(
        len(lookup), start_index, num_scenarios
    )
    if random:
        selected = sample(lookup[start_index:], k=num_scenarios)
    else:
        selected = lookup[start_index:start_index + num_scenarios]
    selected_summary = {}
    selected_mapping = {}
    for scenario in selected:
        selected_summary[scenario] = summaries[scenario]
        selected_mapping[scenario] = os.path.relpath(osp.join(abs_dir_path, mappings[scenario]), output_abs_path)

    save_summary_and_mapping(summary_file, mapping_file, selected_summary, selected_mapping)

    return summaries, mappings
