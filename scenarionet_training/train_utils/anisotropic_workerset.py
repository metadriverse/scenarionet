import copy
import logging
from typing import TypeVar

from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)

# Generic type var for foreach_* methods.
T = TypeVar("T")


@DeveloperAPI
class AnisotropicWorkerSet(WorkerSet):
    """
    Workers are assigned to different scenarios for saving memory/speeding up sampling
    """

    def add_workers(self, num_workers: int) -> None:
        """
        Workers are assigned to different scenarios
        """
        remote_args = {
            "num_cpus": self._remote_config["num_cpus_per_worker"],
            "num_gpus": self._remote_config["num_gpus_per_worker"],
            # memory=0 is an error, but memory=None means no limits.
            "memory": self._remote_config["memory_per_worker"] or None,
            "object_store_memory": self.
                                   _remote_config["object_store_memory_per_worker"] or None,
            "resources": self._remote_config["custom_resources_per_worker"],
        }
        cls = RolloutWorker.as_remote(**remote_args).remote
        for i in range(num_workers):
            config = copy.deepcopy(self._remote_config)
            config["env_config"]["worker_index"] = i
            config["env_config"]["num_workers"] = num_workers
            self._remote_workers.append(self._make_worker(cls, self._env_creator, self._policy_class, i + 1, config))
