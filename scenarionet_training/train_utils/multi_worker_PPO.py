import logging
from typing import Callable, Type

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.env.env_context import EnvContext
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import TrainerConfigDict, \
    EnvType

from scenarionet_training.train_utils.anisotropic_workerset import AnisotropicWorkerSet

logger = logging.getLogger(__name__)


class MultiWorkerPPO(PPOTrainer):
    """
    In this class, each work will have different config for speeding up and saving memory. More importantly, it can
    allow us to cover all test/train cases more evenly
    """

    def _make_workers(self, env_creator: Callable[[EnvContext], EnvType],
                      policy_class: Type[Policy], config: TrainerConfigDict,
                      num_workers: int):
        """Default factory method for a WorkerSet running under this Trainer.

        Override this method by passing a custom `make_workers` into
        `build_trainer`.

        Args:
            env_creator (callable): A function that return and Env given an env
                config.
            policy (Type[Policy]): The Policy class to use for creating the
                policies of the workers.
            config (TrainerConfigDict): The Trainer's config.
            num_workers (int): Number of remote rollout workers to create.
                0 for local only.

        Returns:
            WorkerSet: The created WorkerSet.
        """
        return AnisotropicWorkerSet(
            env_creator=env_creator,
            policy_class=policy_class,
            trainer_config=config,
            num_workers=num_workers,
            logdir=self.logdir)
