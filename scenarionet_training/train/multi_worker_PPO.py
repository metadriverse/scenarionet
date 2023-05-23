from ray.rllib.agents.ppo.ppo import PPOTrainer
import copy
from datetime import datetime
import functools
import logging
import math
import numpy as np
import os
import pickle
import tempfile
import time
from typing import Callable, Dict, List, Optional, Type, Union

import ray
from ray.actor import ActorHandle
from ray.exceptions import RayError
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.normalize_actions import NormalizeActionWrapper
from ray.rllib.env.utils import gym_env_creator
from ray.rllib.evaluation.collectors.simple_list_collector import \
    SimpleListCollector
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import FilterManager, deep_update, merge_dicts
from ray.rllib.utils.annotations import override, PublicAPI, DeveloperAPI
from ray.rllib.utils.deprecation import deprecation_warning, DEPRECATED_VALUE
from ray.rllib.utils.framework import try_import_tf, TensorStructType
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.spaces import space_utils
from ray.rllib.utils.typing import TrainerConfigDict, \
    PartialTrainerConfigDict, EnvInfoDict, ResultDict, EnvType, PolicyID
from ray.tune.logger import Logger, UnifiedLogger
from ray.tune.registry import ENV_CREATOR, register_env, _global_registry
from ray.tune.resources import Resources
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.trainable import Trainable
from ray.tune.trial import ExportFormat
from ray.tune.utils.placement_groups import PlacementGroupFactory

logger = logging.getLogger(__name__)


class MultiWorkerPPO(PPOTrainer):
    @PublicAPI
    def evaluate(self) -> dict:
        """Evaluates current policy under `evaluation_config` settings.

        Note that this default implementation does not do anything beyond
        merging evaluation_config with the normal trainer config.
        """
        # Call the `_before_evaluate` hook.
        self._before_evaluate()

        if self.evaluation_workers is not None:
            # Sync weights to the evaluation WorkerSet.
            self._sync_weights_to_workers(worker_set=self.evaluation_workers)
            self._sync_filters_if_needed(self.evaluation_workers)

        if self.config["custom_eval_function"]:
            logger.info("Running custom eval function {}".format(
                self.config["custom_eval_function"]))
            metrics = self.config["custom_eval_function"](
                self, self.evaluation_workers)
            if not metrics or not isinstance(metrics, dict):
                raise ValueError("Custom eval function must return "
                                 "dict of metrics, got {}.".format(metrics))
        else:
            logger.info("Evaluating current policy for {} episodes.".format(
                self.config["evaluation_num_episodes"]))
            metrics = None
            # No evaluation worker set ->
            # Do evaluation using the local worker. Expect error due to the
            # local worker not having an env.
            if self.evaluation_workers is None:
                try:
                    for _ in range(self.config["evaluation_num_episodes"]):
                        self.workers.local_worker().sample()
                    metrics = collect_metrics(self.workers.local_worker())
                except ValueError as e:
                    if "RolloutWorker has no `input_reader` object" in \
                            e.args[0]:
                        raise ValueError(
                            "Cannot evaluate w/o an evaluation worker set in "
                            "the Trainer or w/o an env on the local worker!\n"
                            "Try one of the following:\n1) Set "
                            "`evaluation_interval` >= 0 to force creating a "
                            "separate evaluation worker set.\n2) Set "
                            "`create_env_on_driver=True` to force the local "
                            "(non-eval) worker to have an environment to "
                            "evaluate on.")
                    else:
                        raise e

            # Evaluation worker set only has local worker.
            elif self.config["evaluation_num_workers"] == 0:
                for _ in range(self.config["evaluation_num_episodes"]):
                    self.evaluation_workers.local_worker().sample()
            # Evaluation worker set has n remote workers.
            else:
                num_rounds = int(
                    math.ceil(self.config["evaluation_num_episodes"] /
                              self.config["evaluation_num_workers"]))
                num_workers = len(self.evaluation_workers.remote_workers())
                num_episodes = num_rounds * num_workers
                for i in range(num_rounds):
                    logger.info("Running round {} of parallel evaluation "
                                "({}/{} episodes)".format(
                        i, (i + 1) * num_workers, num_episodes))
                    ray.get([
                        w.sample.remote()
                        for w in self.evaluation_workers.remote_workers()
                    ])
            if metrics is None:
                metrics = collect_metrics(
                    self.evaluation_workers.local_worker(),
                    self.evaluation_workers.remote_workers())
        return {"evaluation": metrics}
