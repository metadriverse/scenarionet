import json
import shutil
import logging
import os
from typing import List

from metadrive.scenario.scenario_description import ScenarioDescription as SD

from scenarionet.common_utils import save_summary_and_mapping, read_dataset_summary

logger = logging.getLogger(__name__)


class ErrorDescription:
    INDEX = "scenario_index"
    PATH = "file_path"
    FILE_NAME = "file_name"
    ERROR = "error_message"
    METADATA = "metadata"

    @classmethod
    def make(cls, scenario_index, file_path, file_name, error):
        logger.warning(
            "\n Scenario Error, "
            "scenario_index: {}, file_path: {}.\n Error message: {}".format(scenario_index, file_path, str(error))
        )
        return {cls.INDEX: scenario_index, cls.PATH: file_path, cls.FILE_NAME: file_name, cls.ERROR: str(error)}


class ErrorFile:
    PREFIX = "error_scenarios_for"
    DATASET = "dataset_path"
    ERRORS = "errors"

    @classmethod
    def get_error_file_name(cls, dataset_path):
        return "{}_{}.json".format(cls.PREFIX, os.path.basename(dataset_path))

    @classmethod
    def dump(cls, save_dir, errors: List, dataset_path):
        """
        Save test result
        :param save_dir: which dir to save this file
        :param errors: error list, containing a list of dict from ErrorDescription.make()
        :param dataset_path: dataset_path, the dir of dataset_summary.pkl
        """
        file_name = cls.get_error_file_name(dataset_path)
        path = os.path.join(save_dir, file_name)
        with open(path, "w+") as f:
            json.dump({cls.DATASET: dataset_path, cls.ERRORS: errors}, f, indent=4)
        return path

    @classmethod
    def generate_dataset(cls, error_file_path, new_dataset_path, overwrite=False, broken_scenario=False):
        """
        Generate a new database containing all broken scenarios or all good scenarios
        :param error_file_path: error file path
        :param new_dataset_path: a directory where you want to store your data
        :param overwrite: if new_dataset_path exists, whether to overwrite
        :param broken_scenario: generate broken scenarios. You can generate such a broken scenarios for debugging
        :return: database summary, database mapping
        """
        new_dataset_path = os.path.abspath(new_dataset_path)
        if os.path.exists(new_dataset_path):
            if overwrite:
                shutil.rmtree(new_dataset_path)
            else:
                raise ValueError(
                    "Directory: {} already exists! "
                    "Set overwrite=True to overwrite".format(new_dataset_path)
                )
        os.makedirs(new_dataset_path, exist_ok=False)

        with open(error_file_path, "r+") as f:
            error_file = json.load(f)
        origin_dataset_path = error_file[cls.DATASET]
        origin_summary, origin_list, origin_mapping = read_dataset_summary(origin_dataset_path)
        errors = error_file[cls.ERRORS]

        # make new summary
        new_summary = {}
        new_mapping = {}

        new_summary_file_path = os.path.join(new_dataset_path, SD.DATASET.SUMMARY_FILE)
        new_mapping_file_path = os.path.join(new_dataset_path, SD.DATASET.MAPPING_FILE)

        if broken_scenario:
            for error in errors:
                file_name = error[ErrorDescription.FILE_NAME]
                new_summary[file_name] = origin_summary[file_name]
                scenario_dir = os.path.join(origin_dataset_path, origin_mapping[file_name])
                new_mapping[file_name] = os.path.relpath(scenario_dir, new_dataset_path)
        else:
            error_scenario = [error[ErrorDescription.FILE_NAME] for error in errors]
            for scenario in origin_summary:
                if scenario in error_scenario:
                    continue
                new_summary[scenario] = origin_summary[scenario]
                scenario_dir = os.path.join(origin_dataset_path, origin_mapping[scenario])
                new_mapping[scenario] = os.path.relpath(scenario_dir, new_dataset_path)
        save_summary_and_mapping(new_summary_file_path, new_mapping_file_path, new_summary, new_mapping)
        return new_summary, new_mapping
