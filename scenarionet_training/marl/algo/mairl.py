import logging
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Type
import random
import gym
import numpy as np
from gym.spaces import Box
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step, multi_gpu_train_one_step
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.sgd import minibatches, standardized
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.torch_utils import sequence_mask
from ray.rllib.utils.typing import ModelConfigDict, TrainerConfigDict
from ray.util.debug import log_once
from torch.distributions import Normal
from tqdm.auto import tqdm

from newcopo.metadrive_scenario.marl_envs.marl_single_waymo_env import dynamics_parameters_to_embedding
from scenarionet_training.marl.algo.ccppo import CCPPOTrainer, CCPPOPolicy, CCPPOConfig
from scenarionet_training.marl.algo.ccppo import mean_field_ccppo_process, concat_ccppo_process
from scenarionet_training.marl.algo.copo import CoPOModel
from scenarionet_training.marl.utils.env_wrappers import get_lcf_env, get_rllib_compatible_env, get_latent_env, \
    get_latent_posterior_env, LatentEnvBase, LatentPosteriorEnvBase
from metadrive.utils.waymo_utils.waymo_utils import AgentType
from newcopo.metadrive_scenario.marl_envs.marl_waymo_env import MARLWaymoEnv
from newcopo.metadrive_scenario.marl_envs.marl_single_waymo_env import process_expert_trajectory


from scenarionet_training.marl.algo.multihead import MultiheadDynamicsModel

# ModelCatalog.register_custom_model("copo_model", CoPOModel)


torch, nn = try_import_torch()

F = nn.functional

logger = logging.getLogger(__name__)

NEI_REWARDS = "nei_rewards"
NEI_VALUES = "nei_values"
NEI_ADVANTAGE = "nei_advantage"
NEI_TARGET = "nei_target"
LCF_LR = "lcf_lr"
GLOBAL_VALUES = "global_values"
GLOBAL_REWARDS = "global_rewards"
GLOBAL_ADVANTAGES = "global_advantages"
GLOBAL_TARGET = "global_target"
USE_CENTRALIZED_CRITIC = "use_centralized_critic"
CENTRALIZED_CRITIC_OBS = "centralized_critic_obs"
COUNTERFACTUAL = "counterfactual"
USE_DISTRIBUTIONAL_LCF = "use_distributional_lcf"

DYNAMICS_POLICY = "dynamics_policy"

# path to ~/newcopo/newcopo
NEWCOPO_DIR = osp.dirname(osp.dirname(osp.abspath(osp.dirname(__file__))))

DYNAMICS_PARAMETERS_DIM = 5

VEHICLE_STATE_DIM = 269

EXPERT_TRAJECTORY_TOTAL_SIZE = 20 * 8


def _pad_a_traj(traj, max_trajectory_len, should_slice=False):

    if should_slice:
        if traj.shape[1] == 279:
            traj = traj[:, 130:159]
        elif traj.shape[1] == 280:
            traj = traj[:, 130:159]
        elif traj.shape[1] == 269:
            traj = traj[:, 120:149]

    else:
        assert traj.shape[1] == 269

    traj_len = traj.shape[0]
    if traj_len >= max_trajectory_len:
        # sample random index to slice trajectory
        si = random.randint(0, traj_len - max_trajectory_len)
        states = torch.from_numpy(traj[si: si + max_trajectory_len])
        timesteps = torch.arange(start=si, end=si + max_trajectory_len, step=1)
        # all ones since no padding
        traj_mask = torch.ones(max_trajectory_len + 1, dtype=torch.long)  # Add one dim for class token
    else:
        padding_len = max_trajectory_len - traj_len
        # padding with zeros
        states = torch.from_numpy(traj)
        states = torch.cat(
            [states, torch.zeros(([padding_len] + list(states.shape[1:])), dtype=states.dtype)], dim=0
        )
        timesteps = torch.arange(start=0, end=max_trajectory_len, step=1)
        traj_mask = torch.cat(
            [torch.ones(traj_len + 1, dtype=torch.long), torch.zeros(padding_len, dtype=torch.long)], dim=0
        )  # Add one dim for class token
    return states, timesteps, traj_mask


def _pad_a_traj_for_forward_model(batch, max_trajectory_len, latent_dim, device):
    """
    When training the forward model, we don't need to input the latent code.
    """
    traj = batch["obs"]
    next_state = batch["new_obs"]
    actions = batch["actions"]
    dynamics = batch["dynamics_parameters"]
    action_logp = batch["action_logp"]

    actions = action_discrete_to_continuous(actions, 5)

    traj = traj[:, latent_dim:]
    next_state = next_state[:, latent_dim:]
    if traj.shape[1] == VEHICLE_STATE_DIM + 1:
        traj = traj[:, :-1]
        next_state = next_state[:, :-1]
    assert traj.shape[1] == 269
    assert next_state.shape[1] == 269

    traj_len = traj.shape[0]
    if traj_len >= max_trajectory_len:
        # sample random index to slice trajectory
        si = random.randint(0, traj_len - max_trajectory_len)
        states = torch.from_numpy(traj[si: si + max_trajectory_len])
        timesteps = torch.arange(start=si, end=si + max_trajectory_len, step=1)
        # all ones since no padding
        traj_mask = torch.ones(max_trajectory_len * 3, dtype=torch.long)  # Add one dim for class token

        actions = torch.from_numpy(actions[si: si + max_trajectory_len])
        dynamics = torch.from_numpy(dynamics[si: si + max_trajectory_len])
        next_state = torch.from_numpy(next_state[si: si + max_trajectory_len])
        action_logp = torch.from_numpy(action_logp[si: si + max_trajectory_len])

    else:
        padding_len = max_trajectory_len - traj_len
        # padding with zeros
        states = torch.from_numpy(traj)
        states = torch.cat(
            [states, torch.zeros(([padding_len] + list(states.shape[1:])), dtype=states.dtype)], dim=0
        )
        timesteps = torch.arange(start=0, end=max_trajectory_len, step=1)
        traj_mask = torch.cat(
            [torch.ones(traj_len * 3, dtype=torch.long), torch.zeros(padding_len * 3, dtype=torch.long)], dim=0
        )  # Add one dim for class token

        actions = torch.from_numpy(actions)
        actions = torch.cat(
            [actions, torch.zeros(([padding_len] + list(actions.shape[1:])), dtype=actions.dtype)], dim=0
        )

        dynamics = torch.from_numpy(dynamics)
        dynamics = torch.cat(
            [dynamics, torch.zeros(([padding_len] + list(dynamics.shape[1:])), dtype=dynamics.dtype)], dim=0
        )

        next_state = torch.from_numpy(next_state)
        next_state = torch.cat(
            [next_state, torch.zeros(([padding_len] + list(next_state.shape[1:])), dtype=next_state.dtype)], dim=0
        )

        action_logp = torch.from_numpy(action_logp)
        action_logp = torch.cat(
            [action_logp, torch.zeros(([padding_len] + list(action_logp.shape[1:])), dtype=action_logp.dtype)], dim=0
        )


    return (
        states.to(device),
        timesteps.to(device),
        traj_mask.to(device),
        actions.to(device),
        dynamics.to(device),
        next_state.to(device),
        action_logp.to(device),
    )


def split_batch_to_episodes(batch):
    seed_list = np.array([i["environment_seed"] for i in batch["infos"]])
    vid_list = np.array([i["vehicle_id"] for i in batch["infos"]])
    indices = np.arange(len(seed_list))
    unique_seed = np.unique(seed_list)
    episode_dict = {}
    for seed in unique_seed:
        ind = indices[seed_list == seed]
        vid_list_in_this_seed = vid_list[ind]
        for vid in np.unique(vid_list_in_this_seed):
            ep_ind = indices[(seed_list == seed) & (vid_list == vid)]
            episode = SampleBatch({k: v[ep_ind] for k, v in batch.items()})
            unique_ep_id = seed * 100000 + (int(vid) if vid != "sdc" else 0)
            unique_ep_id = int(unique_ep_id)
            episode["unique"] = np.ones([len(episode["obs"]),], dtype=int) * int(unique_ep_id)
            episode_dict[int(unique_ep_id)] = episode
    return episode_dict



def _get_action(value, max_value):
    """Transform a discrete value: [0, max_value) into a continuous value [-1, 1]"""
    action = value / (max_value - 1)
    action = action * 2 - 1
    return action


def action_discrete_to_continuous(action, discrete_action_dim):
    new_action = {}
    # for k, v in action.items():
    assert (0 <= action).all()
    assert (action < discrete_action_dim * discrete_action_dim).all()
    a0 = action % discrete_action_dim
    a0 = _get_action(a0, discrete_action_dim)
    a1 = action // discrete_action_dim
    a1 = _get_action(a1, discrete_action_dim)
    # new_action[k] = [a0, a1]
    # action = new_action
    ret = np.stack([a0, a1], axis=1).astype(np.float32)
    return ret


class NewConfig(CCPPOConfig):
    # TODO: This is identical to the CoPO config! Change if needed
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or CCPPOTrainer)

        # PPO
        self.num_sgd_iter = 10
        self.lr = 1e-4

        # CoPO
        self.enable_copo = True
        self.counterfactual = False
        self.initial_lcf_std = 0.1
        self.lcf_sgd_minibatch_size = None
        self.lcf_num_iters = 5
        self.lcf_lr = 1e-4
        self.use_distributional_lcf = True
        self.use_centralized_critic = False
        self.fuse_mode = "none"
        self.update_from_dict({
            "model": {
                "custom_model": "policy_model",
            }})

        # Discriminator
        self.enable_discriminator = True  # Do we really need this?
        self.discriminator_reward_native = True
        self.discriminator_add_action = False
        self.discriminator_use_tanh = False
        self.discriminator_l2 = 1e-5
        self.discriminator_lr = 1e-4
        self.discriminator_num_iters = 5
        self.discriminator_sgd_minibatch_size = 1024

        # Switch algorithm
        self.algorithm = None,  # Must in ["gail", "airl"]

        # Inverse Model and Forward Model parameters
        self.inverse_num_iters = 100
        self.inverse_sgd_minibatch_size = 512
        self.inverse_lr = 1e-4

        # Dataset
        self.use_copo_dataset = False
        # self.use_copo_dataset_with_inverse_model = False  # Set these two = True to use inverse model for CoPO dataset

        # Dynamics and Forward Model
        self.randomized_dynamics = None  # Must in [None, "mean", "std", "nn", "nnstd", "gmm", "dynamics_policy"]
        self.dynamics_num_iters = 5
        self.dynamics_lr = 1e-5
        self.dynamics_posterior_weight = 0.0
        self.dynamics_gmm_k = 3

        # Posterior model (the latent model)
        self.posterior_lr = 0.0
        self.posterior_num_samples = 1  # TODO: Do we tune this hyper?
        self.posterior_temperature = 1.0
        self.posterior_match_loss_weight = 0.0
        self.num_neg_samples = 1

        # Reward augmentation
        self.reward_augmentation_weight = 0

        # Latent variable
        self.enable_latent = False
        self.enable_latent_posterior = False
        self.latent_dim = 10
        # self.latent_trajectory_len = -1  # Deprecated
        self.trajectory_len = 20  # Set this to >= 100 would be better

        self.debug = False


    def validate(self):
        # Update these first since they affect obs space!
        self["env_config"]["enable_copo"] = self["enable_copo"]
        self["env_config"]["enable_latent"] = self["enable_latent"]
        self["env_config"]["latent_dim"] = self["latent_dim"]
        # self["env_config"]["latent_trajectory_len"] = self["latent_trajectory_len"]

        super().validate()

        assert self[USE_DISTRIBUTIONAL_LCF]
        if self[USE_DISTRIBUTIONAL_LCF]:
            self.update_from_dict({
                "env_config": {
                    "return_native_reward": True,
                    "lcf_dist": "normal",
                    "lcf_normal_std": self["initial_lcf_std"]
                }})
            self.model["custom_model_config"][USE_DISTRIBUTIONAL_LCF] = self[USE_DISTRIBUTIONAL_LCF]
            self.model["custom_model_config"]["initial_lcf_std"] = self["initial_lcf_std"]

        self.model["custom_model_config"]["algorithm"] = self["algorithm"]
        self.model["custom_model_config"]["latent_dim"] = self["latent_dim"]
        self.model["custom_model_config"]["enable_latent"] = self["enable_latent"]
        self.model["custom_model_config"]["enable_latent_posterior"] = self["enable_latent_posterior"]
        self.model["custom_model_config"]["dynamics_posterior_weight"] = self["dynamics_posterior_weight"]
        self.model["custom_model_config"]["posterior_lr"] = self["posterior_lr"]
        self.model["custom_model_config"]["dynamics_gmm_k"] = self["dynamics_gmm_k"]
        self.model["custom_model_config"]["discriminator_add_action"] = self["discriminator_add_action"]
        self.model["custom_model_config"]["trajectory_len"] = self["trajectory_len"]
        # self.model["custom_model_config"]["latent_trajectory_len"] = self["latent_trajectory_len"]

        # I should have a check here as follows.
        # But self.env is a string and is not a class, so I can't do the check.
        # Maybe the user should do the check by themselves. And there will be error of course in the environment
        # if the user makes mistake on using get_latent_env or get_latent_posterior_env. So never mind.
        # if self.enable_latent_posterior:
        #     assert self.enable_latent
        #     assert issubclass(self.env, LatentPosteriorEnvBase)
        # if self.enable_latent:
        #     assert issubclass(self.env, LatentEnvBase)

        # Need to sync the config to the environment side
        self["env_config"]["randomized_dynamics"] = self["randomized_dynamics"]

        if self.reward_augmentation_weight > 0:
            assert self.env_config["driving_reward"] == 0

        assert self["randomized_dynamics"] in [None, "mean", "std", "nn", "nnstd", "gmm", "multihead", DYNAMICS_POLICY]
        self.model["custom_model_config"]["randomized_dynamics"] = self["randomized_dynamics"]
        # self.model["custom_model_config"]["dynamics_use_std"] = self["dynamics_use_std"]

        if "discrete_action_dim" in self.env_config:
            self["discrete_action_dim"] = self.env_config["discrete_action_dim"]
        else:
            self["discrete_action_dim"] = 5

        assert self.algorithm in ["gail", "airl"]


class PolicyModel(CoPOModel):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        # if model_config["custom_model_config"]["enable_latent"]:
        #     obs_dim = int(np.prod(obs_space.shape)) + model_config["custom_model_config"]["latent_dim"]
        # obs_space += self.model_config["latent_dim"]
        # obs_space = Box(obs_space.low[0], obs_space.high[0], (obs_dim, ))
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
    #
    # def forward(self, input_dict, state, seq_lens):
    #     print(11111)
    #     if self.model_config["custom_model_config"]["enable_latent"]:
    #         if isinstance(input_dict["infos"][0], dict):
    #             print(1111111)
    #         else:
    #             pass  # input_dict["obs_flat"] is 269 + 10 when initializing.
    #     ret = super().forward(input_dict=input_dict, state=state, seq_lens=seq_lens)
    #     return ret


ModelCatalog.register_custom_model("policy_model", PolicyModel)


# Credit: https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
def _sample_gumbel(shape, eps=1e-20, device=None):
    U = torch.rand(shape)
    U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def _gumbel_softmax_sample(logits, temperature, device=None):
    y = logits + _sample_gumbel(logits.size(), device=device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, device):
    """
    ST-gumple-softmax
    input: [bs, K, n_class]
    return: flatten --> [bs, n_class] an one-hot vector
    """
    y = _gumbel_softmax_sample(logits, temperature, device=device)
    # y in [bs, K, n_class] in R^[0, 1]

    shape = y.size()
    # shape = [bs, K, n_class]

    _, ind = y.max(dim=-1)
    # ind is in [bs, K], has size bs * K

    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    # y_hard is in [bs * K, n_class]

    y_hard.scatter_(dim=1, index=ind.view(-1, 1), value=1)
    # write 1 in the selected classes

    y_hard = y_hard.view(*shape)

    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    # output is in [bs, K, n_class]

    return y_hard


class Discriminator(TorchModelV2, nn.Module):
    def __init__(
            self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
            model_config: ModelConfigDict, name: str
    ):

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # assert not isinstance(action_space, Box)

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2

        # self._logits = None
        self.algorithm = self.model_config["custom_model_config"]["algorithm"]
        self.latent_dim = self.model_config["custom_model_config"]["latent_dim"]
        self.enable_latent = self.model_config["custom_model_config"]["enable_latent"]
        self.enable_latent_posterior = self.model_config["custom_model_config"]["enable_latent_posterior"]
        if self.enable_latent_posterior:
            assert self.enable_latent

        # ========== Our Modification: We compute the centralized critic obs size here! ==========

        if self.algorithm in ["airl"]:
            # === Reward Model ===
            if self.enable_latent:
                obs_dim = 269 + self.latent_dim
            else:
                obs_dim = 269
            prev_layer_size = obs_dim
            if self.model_config["custom_model_config"]["discriminator_add_action"]:
                prev_layer_size += 2
            layers = []
            for size in hiddens:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=size,
                        initializer=normc_initializer(1.0),
                        activation_fn="relu"
                    )
                )
                prev_layer_size = size
            layers.append(SlimFC(
                in_size=prev_layer_size,
                out_size=1,
                initializer=normc_initializer(0.01),
                activation_fn=None
            ))
            self._reward_model = nn.Sequential(*layers)

            # === Value Model ===
            layers = []
            if self.enable_latent:
                obs_dim = 269 + self.latent_dim
            else:
                obs_dim = 269
            prev_layer_size = obs_dim  # Hardcoded!!!
            for size in hiddens:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=size,
                        initializer=normc_initializer(1.0),
                        activation_fn="relu"
                    )
                )
                prev_layer_size = size
            layers.append(SlimFC(
                in_size=prev_layer_size,
                out_size=1,
                initializer=normc_initializer(0.01),
                activation_fn=None
            ))
            self._value_model = nn.Sequential(*layers)

            # === Inverse Dynamic Model ===
            layers = []
            if self.enable_latent:
                obs_dim = (269 + self.latent_dim) * 2
            else:
                obs_dim = 269 * 2
            prev_layer_size = obs_dim
            # prev_layer_size = 269 * 2 + self.latent_dim
            # prev_layer_size = 150 * 2
            inverse_hiddens = [256, 256]
            for size in inverse_hiddens:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=size,
                        initializer=normc_initializer(1.0),
                        activation_fn=activation
                    )
                )
                prev_layer_size = size
            layers.append(SlimFC(
                in_size=prev_layer_size,
                # out_size=int(np.prod(action_space.shape)),
                out_size=int(np.prod(action_space.shape)) if isinstance(action_space, Box) else action_space.n,
                initializer=normc_initializer(0.01),
                activation_fn="tanh" if self.model_config["discriminator_use_tanh"] else None
            ))
            self._inverse_model = nn.Sequential(*layers)

            if isinstance(self.action_space, Box):
                self._inverse_loss = torch.nn.MSELoss()
            else:
                self._inverse_loss = torch.nn.CrossEntropyLoss()

        elif self.algorithm in ["gail"]:
            # === Discriminator ===
            self._gail_discriminator_feature = nn.Sequential(
                SlimFC(
                    in_size=VEHICLE_STATE_DIM if not self.enable_latent else VEHICLE_STATE_DIM + self.latent_dim,
                    out_size=256,
                    initializer=normc_initializer(1.0),
                    activation_fn="relu"
                ),
                SlimFC(
                    in_size=256,
                    out_size=128,
                    initializer=normc_initializer(1.0),
                    activation_fn=None
                )
            )

            # if self.enable_latent:
            #     obs_dim = (269 + self.latent_dim) * 2
            # else:
            #     obs_dim = 269 * 2
            prev_layer_size = 128 * 2
            layers = []
            for size in [256, ]:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=size,
                        initializer=normc_initializer(1.0),
                        activation_fn="relu"
                    )
                )
                prev_layer_size = size
            layers.append(SlimFC(
                in_size=prev_layer_size,
                out_size=1,
                initializer=normc_initializer(0.01),
                activation_fn=None
            ))
            self._gail_discriminator = nn.Sequential(*layers)
            self._gail_discriminator_loss = nn.BCELoss()

        else:
            raise ValueError()

        self.randomized_dynamics = self.model_config["custom_model_config"]["randomized_dynamics"]
        if self.randomized_dynamics:


            # === Forward Model ===
            # if self.enable_latent:
            #     prev_layer_size = VEHICLE_STATE_DIM + 2 + DYNAMICS_PARAMETERS_DIM + self.latent_dim
            # else:
            #     prev_layer_size = VEHICLE_STATE_DIM + 2 + DYNAMICS_PARAMETERS_DIM
            # layers = []
            # for size in [256, 256]:
            #     layers.append(
            #         SlimFC(
            #             in_size=prev_layer_size,
            #             out_size=size,
            #             initializer=normc_initializer(1.0),
            #             activation_fn="relu"
            #         )
            #     )
            #     prev_layer_size = size
            # layers.append(SlimFC(
            #     in_size=prev_layer_size,
            #     out_size=VEHICLE_STATE_DIM,  # Predict the s_t+1 size
            #     initializer=normc_initializer(0.01),
            #     activation_fn=None
            # ))
            # self._forward_model = nn.Sequential(*layers)

            # Use Transformer to be the forward model
            from scenarionet_training.marl.algo.decision_transformer import ForwardModelTransformer
            self._forward_model = ForwardModelTransformer(
                state_dim=VEHICLE_STATE_DIM,
                output_dim=VEHICLE_STATE_DIM,  # The latent state representation by _gail_discriminator_feature
                n_blocks=2,
                h_dim=64,
                context_len=self.model_config["custom_model_config"]["trajectory_len"] * 3,
                n_heads=1,
                drop_p=0.1,  # Dropout
                max_timestep=1000
                # Note: This is used for create the embedding of timestep. Original 4090. 200 In our case. Set to 1000 for fun.
            )

            self._forward_model_loss = torch.nn.MSELoss()
            if self.randomized_dynamics == "std":
                self.dynamics_parameters = torch.nn.Parameter(
                    torch.cat([torch.zeros([DYNAMICS_PARAMETERS_DIM]), torch.zeros([DYNAMICS_PARAMETERS_DIM]) - 2],
                              dim=0),
                    requires_grad=True
                )
            elif self.randomized_dynamics == "mean":
                self.dynamics_parameters = torch.nn.Parameter(torch.zeros([DYNAMICS_PARAMETERS_DIM]), requires_grad=True)
            elif self.randomized_dynamics in ["nn", "nnstd", DYNAMICS_POLICY]:
                # === Dynamics Model ===
                layers = []
                prev_layer_size = self.latent_dim
                for size in [256, 256]:
                    layers.append(
                        SlimFC(
                            in_size=prev_layer_size,
                            out_size=size,
                            initializer=normc_initializer(1.0),
                            activation_fn="relu"
                        )
                    )
                    prev_layer_size = size
                layers.append(SlimFC(
                    in_size=prev_layer_size,
                    out_size=DYNAMICS_PARAMETERS_DIM if self.randomized_dynamics == "nn" else DYNAMICS_PARAMETERS_DIM * 2,
                    initializer=normc_initializer(0.01),
                    activation_fn=None  # Scale to [-1, 1]
                ))
                self._dynamics_model = nn.Sequential(*layers)
            elif self.randomized_dynamics in ["gmm"]:
                # === Dynamics Model ===
                layers = []
                prev_layer_size = self.latent_dim
                for size in [256, 256]:
                    layers.append(
                        SlimFC(
                            in_size=prev_layer_size,
                            out_size=size,
                            initializer=normc_initializer(1.0),
                            activation_fn="relu"
                        )
                    )
                    prev_layer_size = size
                layers.append(SlimFC(
                    in_size=prev_layer_size,
                    out_size=self.model_config["custom_model_config"]["dynamics_gmm_k"],
                    initializer=normc_initializer(0.01),
                    activation_fn=None  # Scale to [-1, 1]
                ))
                self._dynamics_model = nn.Sequential(*layers)
                # === Dynamics Model ===
                # TODO: Maybe we can add covariance matrix for each Normal distribution component.
                dy_param = torch.zeros([
                    self.model_config["custom_model_config"]["dynamics_gmm_k"], DYNAMICS_PARAMETERS_DIM * 2
                ])
                dy_param[:, DYNAMICS_PARAMETERS_DIM:] -= 2
                self.dynamics_parameters = torch.nn.Parameter(dy_param, requires_grad=True)

                layers = []
                prev_layer_size = DYNAMICS_PARAMETERS_DIM
                for size in [256, 256]:
                    layers.append(
                        SlimFC(
                            in_size=prev_layer_size,
                            out_size=size,
                            initializer=normc_initializer(1.0),
                            activation_fn="relu"
                        )
                    )
                    prev_layer_size = size
                layers.append(SlimFC(
                    in_size=prev_layer_size,
                    out_size=self.latent_dim * 2,
                    initializer=normc_initializer(0.01),
                    activation_fn=None  # Scale to [-1, 1]
                ))
                self._dynamics_model_reverse = nn.Sequential(*layers)
            elif self.randomized_dynamics in ["multihead"]:
                # self._dynamics_model = MultiheadDynamicsModel(
                #     K=self.model_config["custom_model_config"]["dynamics_gmm_k"],
                #     output_dim=DYNAMICS_PARAMETERS_DIM
                # )
                # === Dynamics Model ===
                # TODO: Maybe we can add covariance matrix for each Normal distribution component.
                dy_param = torch.zeros([
                    self.model_config["custom_model_config"]["dynamics_gmm_k"], DYNAMICS_PARAMETERS_DIM * 2
                ])
                dy_param[:, DYNAMICS_PARAMETERS_DIM:] -= 2
                self.dynamics_parameters = torch.nn.Parameter(dy_param, requires_grad=True)

            else:
                raise ValueError()

            # # === Dynamics Model ===
            # layers = []
            # prev_layer_size = DYNAMICS_PARAMETERS_DIM
            # for size in [256, 256]:
            #     layers.append(
            #         SlimFC(
            #             in_size=prev_layer_size,
            #             out_size=size,
            #             initializer=normc_initializer(1.0),
            #             activation_fn="relu"
            #         )
            #     )
            #     prev_layer_size = size
            # layers.append(SlimFC(
            #     in_size=prev_layer_size,
            #     out_size=self.latent_dim,
            #     initializer=normc_initializer(0.01),
            #     activation_fn=None  # Scale to [-1, 1]
            # ))
            # self._dynamics_recon_model = nn.Sequential(*layers)
            # # === Dynamics Model ===
            # # TODO: Maybe we can add covariance matrix for each Normal distribution component.
            # dy_param = torch.zeros([
            #     self.model_config["custom_model_config"]["dynamics_gmm_k"], DYNAMICS_PARAMETERS_DIM * 2
            # ])
            # dy_param[:, DYNAMICS_PARAMETERS_DIM:] -= 2
            # self.dynamics_parameters = torch.nn.Parameter(dy_param, requires_grad=True)

        if self.enable_latent_posterior:
            # === Posterior Model ===
            # The input to this model is a chunk of expert trajectory in the dataset.
            # prev_layer_size = EXPERT_TRAJECTORY_TOTAL_SIZE  # Hardcoded expert trajectory length

            # 1st layer: 256, no BN
            # 2nd layer: 256, no BN
            # 3rd layer: 256, with BN
            # 4th layer: 256, with BN
            # --- We now have a feature with 256 ---
            # Output head: 256 -> latent_dim * 2
            # Prediction head:
            #   1st: 256 -> 256, with BN, ReLU
            #   2nd: 256 -> 256

            # # prev_layer_size = 2 * 269 + 2  # Hardcoded expert trajectory length + width/height
            # prev_layer_size = self.model_config["custom_model_config"]["trajectory_len"] * 29
            # layers = []
            # for size in [256, 256]:
            #     layers.append(
            #         SlimFC(
            #             in_size=prev_layer_size,
            #             out_size=size,
            #             initializer=normc_initializer(1.0),
            #             activation_fn="relu"
            #         )
            #     )
            #     prev_layer_size = size
            #
            # layers.extend([
            #     # nn.Linear(256, 256, bias=False),
            #     # nn.BatchNorm1d(256),
            #     # nn.ReLU(inplace=True),  # first layer
            #     # nn.Linear(256, 256, bias=False),
            #     # nn.BatchNorm1d(256),
            #     # nn.ReLU(inplace=True),  # second layer
            #     # nn.Linear(256, 256, bias=False),
            #     # nn.BatchNorm1d(256, affine=False),
            #     nn.Linear(256, self.latent_dim * 2)
            # ])
            # # layers.append(SlimFC(
            # #     in_size=prev_layer_size,
            # #     out_size=self.latent_dim * 2,  # Predict the s_t+1 size
            # #     initializer=normc_initializer(0.01),
            # #     activation_fn=None
            # # ))
            # self._posterior_model = nn.Sequential(*layers)

            from scenarionet_training.marl.algo.decision_transformer import DecisionTransformer
            self._posterior_model = DecisionTransformer(
                state_dim=VEHICLE_STATE_DIM,  # 29 in our case
                output_dim=self.latent_dim,  # 0 in our case. Set to 2 for testing.
                n_blocks=2,
                h_dim=64,
                context_len=self.model_config["custom_model_config"]["trajectory_len"],
                n_heads=1,
                drop_p=0.1,  # Dropout
                max_timestep=1000
                # Note: This is used for create the embedding of timestep. Original 4090. 200 In our case. Set to 1000 for fun.
            )
            self._posterior_model_target = DecisionTransformer(
                state_dim=VEHICLE_STATE_DIM,  # 29 in our case
                output_dim=self.latent_dim,  # 0 in our case. Set to 2 for testing.
                n_blocks=2,
                h_dim=64,
                context_len=self.model_config["custom_model_config"]["trajectory_len"],
                n_heads=1,
                drop_p=0.1,  # Dropout
                max_timestep=1000
                # Note: This is used for create the embedding of timestep. Original 4090. 200 In our case. Set to 1000 for fun.
            )

            # prev_layer_size = 2 * 269 + 2  # Hardcoded expert trajectory length + width/height
            # layers = []
            # for size in [256, 256]:
            #     layers.append(
            #         SlimFC(
            #             in_size=prev_layer_size,
            #             out_size=size,
            #             initializer=normc_initializer(1.0),
            #             activation_fn="relu"
            #         )
            #     )
            #     prev_layer_size = size
            # layers.extend([
            #     # nn.Linear(256, 256, bias=False),
            #     # nn.BatchNorm1d(256),
            #     # nn.ReLU(inplace=True),  # first layer
            #     # nn.Linear(256, 256, bias=False),
            #     # nn.BatchNorm1d(256, affine=False),
            #     nn.Linear(256, self.latent_dim * 2)
            # ])
            # self._posterior_model_target =  nn.Sequential(*layers)
            # self._posterior_model_target.load_state_dict(self._posterior_model.state_dict())
            # self._posterior_model_head = nn.Linear(256, self.latent_dim * 2)

            # build a 3-layer projector
            # self._posterior_model_projector = nn.Sequential(
            #     nn.Linear(256, 256, bias=False),
            #     nn.BatchNorm1d(256),
            #     nn.ReLU(inplace=True),  # first layer
            #     # nn.Linear(256, 256, bias=False),
            #     # nn.BatchNorm1d(256),
            #     # nn.ReLU(inplace=True),  # second layer
            #     nn.Linear(256, 256, bias=False),
            #     nn.BatchNorm1d(256, affine=False)
            # )  # output layer
            # self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

            # build a 2-layer predictor
            # self._posterior_model_prediction = nn.Sequential(
            #     nn.Linear(256, 256, bias=False),
            #     nn.BatchNorm1d(256),
            #     nn.ReLU(inplace=True),  # hidden layer
            #     nn.Linear(256, 256)
            # )  # output layer


            # Copy the posterior model
            # TODO: Temporarily stop using the target model for posterior model.
            # prev_layer_size = 2 * 269 + 2  # Hardcoded expert trajectory length + width/height
            # layers = []
            # for size in [256, 256]:
            #     layers.append(
            #         SlimFC(
            #             in_size=prev_layer_size,
            #             out_size=size,
            #             initializer=normc_initializer(1.0),
            #             activation_fn="relu"
            #         )
            #     )
            #     prev_layer_size = size
            # layers.append(SlimFC(
            #     in_size=prev_layer_size,
            #     out_size=self.latent_dim * 2,  # Predict the s_t+1 size
            #     initializer=normc_initializer(0.01),
            #     activation_fn=None
            # ))
            # self._posterior_target_model = nn.Sequential(*layers)
            # self._posterior_target_model.load_state_dict(self._posterior_model.state_dict())  # Sync

            # prev_layer_size = self.latent_dim + 269 # Hardcoded expert trajectory length
            # layers = []
            # for size in [256, 256]:
            #     layers.append(
            #         SlimFC(
            #             in_size=prev_layer_size,
            #             out_size=size,
            #             initializer=normc_initializer(1.0),
            #             activation_fn="relu"
            #         )
            #     )
            #     prev_layer_size = size
            # layers.append(SlimFC(
            #     in_size=prev_layer_size,
            #     out_size=269,  # Predict expert next state
            #     initializer=normc_initializer(0.01),
            #     activation_fn=None
            # ))
            # self._posterior_model_recon = nn.Sequential(*layers)

    @torch.no_grad()
    def update_posterior_target(self, tau = 0.995):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self._posterior_model.parameters(), self._posterior_model_target.parameters()):
            param_k.data = param_k.data * tau + param_q.data * (1. - tau)

    def get_forward_parameters(self):
        return list(self._forward_model.parameters())

    def get_dynamics_parameters(self):
        if self.randomized_dynamics in ["mean", "std"]:
            return [self.dynamics_parameters]
        elif self.randomized_dynamics in ["nn", "nnstd", DYNAMICS_POLICY]:
            return list(self._dynamics_model.parameters())
        elif self.randomized_dynamics in ["gmm"]:
            return [self.dynamics_parameters] + list(self._dynamics_model.parameters()) + list(self._dynamics_model_reverse.parameters())
        elif self.randomized_dynamics in ["multihead"]:
            return self.dynamics_parameters
        else:
            raise ValueError()

    def get_posterior_parameters(self):
        return list(self._posterior_model.parameters()) # + list(self._posterior_model_head.parameters())
            # list(self._posterior_model_prediction.parameters())

    def get_dynamics_parameters_distribution(self, nn_batch_size=500, latent=None, device=None):
        if self.randomized_dynamics == "std":
            loc, log_std = torch.chunk(self.dynamics_parameters, 2, dim=0)
            return loc, log_std

        elif self.randomized_dynamics == "gmm":
            # mean_list = []
            # log_std_list = []
            # dy_params = torch.chunk(self.dynamics_parameters,
            #                         self.model_config["custom_model_config"]["dynamics_gmm_k"], dim=0)
            # for dy_param in dy_params:
            #     dy_param = dy_param.reshape(-1)
            #     loc, log_std = torch.chunk(dy_param, 2, dim=0)
            #     mean_list.append(loc)
            #     log_std_list.append(log_std)
            # Call dynamics model to get the weight
            if self.enable_latent:
                assert latent is not None
                assert latent.shape[0] == nn_batch_size
                noise = latent
            else:
                noise = torch.normal(0, 1, size=[nn_batch_size, self.latent_dim]).to(device)
            dynamics_parameters = self._dynamics_model(noise)
            # return mean_list, log_std_list, dynamics_parameters
            return self.dynamics_parameters, dynamics_parameters

        elif self.randomized_dynamics == "mean":
            # We make log std -> -100, such that std -> 0
            return self.dynamics_parameters, torch.zeros([DYNAMICS_PARAMETERS_DIM]) - 100

        elif self.randomized_dynamics == "nn":
            if self.enable_latent:
                assert latent is not None
                assert latent.shape[0] == nn_batch_size
                noise = latent
            else:
                noise = torch.normal(0, 1, size=[nn_batch_size, self.latent_dim]).to(device)
            dynamics_parameters = self._dynamics_model(noise)
            # in [-1, 1]

            return dynamics_parameters

        elif self.randomized_dynamics in ["nnstd", "dynamics_policy"]:
            if self.enable_latent:
                assert latent is not None
                assert latent.shape[0] == nn_batch_size
                noise = latent
            else:
                noise = torch.normal(0, 1, size=[nn_batch_size, self.latent_dim]).to(device)
            dynamics_parameters = self._dynamics_model(noise)
            # in [-1, 1]
            loc, log_std = torch.chunk(dynamics_parameters, 2, dim=1)

            std = torch.exp(log_std.clamp(-20, 10))
            dist = Normal(loc, std)
            dynamics_parameters = dist.rsample()

            return dynamics_parameters

        elif self.randomized_dynamics == "multihead":
            return self.dynamics_parameters

        else:
            raise ValueError()

    def get_discriminator_parameters(self):

        if self.algorithm in ["airl"]:
            return list(self._reward_model.parameters()) + list(self._value_model.parameters())

        elif self.algorithm in ["gail"]:
            return list(self._gail_discriminator.parameters()) + list(self._gail_discriminator_feature.parameters())

        else:
            raise ValueError()

    def get_inverse_parameters(self):
        return list(self._inverse_model.parameters())

    def get_internal_reward(self, *, obs, act, latent=None):
        assert self.algorithm in ["airl"]

        if obs.shape[1] == 270:
            obs = obs[:, :-1]

        if self.model_config["custom_model_config"]["discriminator_add_action"]:
            rew_input = torch.cat([obs, act], dim=-1)
        else:
            rew_input = obs

        # Note: Latent is in obs already! So we don't need it!
        # if self.enable_latent:
        #     rew_input = torch.cat([latent, rew_input], dim=-1)

        r = self._reward_model(rew_input)
        r = r.reshape(-1)
        return r

    def get_reward(self, *, obs, act=None, next_obs=None, action_logp=None, latent=None,
                   discriminator_reward_native=False, gamma=None):

        if self.algorithm in ["airl"]:

            if not discriminator_reward_native:
                # Just return the reward if not using native AIRL
                rew = self.get_internal_reward(obs=obs, act=act, latent=latent)
                return rew

            log_p_tau, log_q_tau, log_pq, reward, value, next_value = self._get_airl_logits(
                obs=obs, act=act, next_obs=next_obs, action_logp=action_logp, gamma=gamma,
                latent=latent
            )

            scores = torch.exp(log_p_tau - log_pq)

            # scores should be in [0, 1]
            scores = torch.clamp(scores, 1e-6, 1 - 1e-6)
            ret = torch.log(scores) - torch.log(1 - scores)

            return ret

        elif self.algorithm in ["gail"]:
            if obs.shape[1] == 270:
                obs = obs[:, :-1]
            if next_obs.shape[1] == 270:
                next_obs = next_obs[:, :-1]

            feat1 = self._gail_discriminator_feature(obs)
            feat2 = self._gail_discriminator_feature(next_obs)
            ret = self._gail_discriminator(torch.cat([feat1, feat2], dim=-1))
            ret = nn.functional.sigmoid(ret)
            ret = ret.reshape(-1)
            return ret

        else:
            raise ValueError()

    def get_value(self, obs, latent):

        if obs.shape[1] == 270:
            obs = obs[:, :-1]

        # Note: Latent is in obs already! So we don't need it!
        # if self.enable_latent:
        #     obs = torch.cat([latent, obs], dim=-1)

        ret = self._value_model(obs)
        ret = ret.reshape(-1)
        # print("Return value: ", ret.shape)
        return ret


    # def get_latent_distribution(self, expert_traj):
    #     ret = self._posterior_model(expert_traj)  # [latent_dim, ]
    #     loc, log_std = torch.chunk(ret, 2, dim=-1)
    #     return loc, log_std

    # def get_latent_distribution(self, state, next_state, carsize, allow_train=False):
    # def get_latent_distribution(self, timesteps, state, next_state=None, carsize=None, allow_train=False):
    #     # if state.shape[1] == 270:
    #     #     state = state[:, :-1]
    #     # if next_state.shape[1] == 270:
    #     #     next_state = next_state[:, :-1]
    #     # assert state.shape[1] == 269
    #     # assert next_state.shape[1] == 269
    #     # assert carsize.shape[1] == 2
    #
    #     # assert len(state) == 3
    #
    #     if allow_train:
    #         self._posterior_model.train()
    #         # feat = self._posterior_model(torch.cat([state, next_state, carsize], dim=-1))  # output a 256 feature
    #         feat = self._posterior_model(timesteps, state)  # output a 256 feature
    #     else:
    #         with torch.no_grad():
    #             self._posterior_model.eval()
    #             # feat = self._posterior_model(torch.cat([state, next_state, carsize], dim=-1))  # output a 256 feature
    #             feat = self._posterior_model(timesteps, state)  # output a 256 feature
    #             self._posterior_model.train()
    #             # print("_posterior_model: ", list(self._posterior_model.state_dict().values())[0][:10, :10])
    #
    #     # ret = self._posterior_model_head(feat)  # 256 -> latent_dim * 2
    #     # loc, log_std = torch.chunk(ret, 2, dim=-1)
    #     loc, log_std = torch.chunk(feat, 2, dim=-1)
    #     return loc, log_std


    # def get_latent(self, expert_traj):
    # def get_latent(self, state, next_state, carsize, allow_train=False):
    # def get_latent(self, timesteps, state, next_state=None, carsize=None, allow_train=False):
    #     # d_mean, d_log_std = self.get_latent_distribution(state, next_state, carsize, allow_train=allow_train)
    #     d_mean, d_log_std = self.get_latent_distribution(timesteps, state, allow_train=allow_train)
    #     d_std = torch.exp(d_log_std.clamp(-20, 10))
    #     dist = Normal(d_mean, d_std)
    #     latent = dist.rsample()
    #     return latent
    def get_latent(self, timesteps, state, mask, use_target=False):
        if use_target:
            latent = self._posterior_model_target(timesteps, state, mask)
        else:
            latent = self._posterior_model(timesteps, state, mask)
        # latent = torch.nn.functional.normalize(latent, dim=-1)
        return latent

    # def get_latent_projection_and_prediction(self, state, next_state, carsize, use_target=False):
    def get_latent_projection_and_prediction(self, timesteps, state, mask, next_state=None, carsize=None, use_target=False):
        # if state.shape[1] == 270:
        #     state = state[:, :-1]
        # if next_state.shape[1] == 270:
        #     next_state = next_state[:, :-1]
        # assert state.shape[1] == 269
        # assert next_state.shape[1] == 269
        # assert carsize.shape[1] == 2
        # embedding = self._posterior_model(torch.cat([state, next_state, carsize], dim=-1))  # output a 256 feature
        # pred = self._posterior_model_prediction(embedding)  # output a 256 feature
        # return embedding, pred

        # TODO: Target network is temporarily removed.
        # if use_target:
        #     with torch.no_grad():
        #         feat = self._posterior_model_target(torch.cat([state, next_state, carsize], dim=-1))  # output a 256 feature
                # feat = self._posterior_model_target(state)  # output a 256 feature

        # else:
            # feat = self._posterior_model(torch.cat([state, next_state, carsize], dim=-1))  # output a 256 feature
        feat = self._posterior_model(timesteps, state, mask)  # output a 256 feature
        # ret = self._posterior_model_head(feat)  # 256 -> latent_dim * 2
        # Don't chunk
        return feat

    def compute_inverse_loss(self, obs, next_obs, actions, latent, low, high, dynamics_parameters):

        # if self.enable_latent:
        # obs = obs[:, self.latent_dim:]
        # next_obs = next_obs[:, self.latent_dim:]

        pseudo_action = self.compute_pseudo_action(obs, next_obs, latent=latent, low=low, high=high,
                                                   dynamics_parameters=dynamics_parameters)
        if isinstance(self.action_space, Box):
            inv_loss = self._inverse_loss(input=pseudo_action, target=actions.float())
        else:
            inv_loss = self._inverse_loss(input=pseudo_action, target=actions.long())
        return inv_loss, pseudo_action

    def compute_pseudo_action(self, obs, next_obs, latent, low, high, dynamics_parameters):
        """
        The inverse dynamics model should take raw obs and next obs (without latent in it) as input,
        and also a latent variable to implicitly specify the dynamics
        """
        # inv_input = torch.cat([obs[:, :150], next_obs[:, :150]], dim=-1)

        if obs.shape[1] == 270:
            obs = obs[:, :-1]

        if next_obs.shape[1] == 270:
            next_obs = next_obs[:, :-1]

        if self.enable_latent:
            assert obs.shape[1] == next_obs.shape[1] == 269 + self.latent_dim
        else:
            assert obs.shape[1] == next_obs.shape[1] == 269

        if self.enable_latent:
            #     obs = obs[:, self.latent_dim:]
            #     next_obs = next_obs[:, self.latent_dim:]
            inv_input = torch.cat([obs, next_obs], dim=-1)
        else:
            inv_input = torch.cat([obs, next_obs], dim=-1)

        # Note: Latent is in obs already! So we don't need it!
        # if self.enable_latent:
        #     inv_input = torch.cat([latent, inv_input], dim=-1)

        pseudo_action = self._inverse_model(inv_input)

        if self.model_config["discriminator_use_tanh"]:
            raise ValueError()
            # pseudo_action = (pseudo_action + 1) / 2 * (high - low) + low
            # assert pseudo_action.max().item() <= high
            # assert pseudo_action.min().item() >= low

        return pseudo_action

    def _get_airl_logits(self, *, obs, act, next_obs, action_logp, latent, gamma):
        log_q_tau = action_logp
        reward = self.get_internal_reward(obs=obs, act=act, latent=latent)

        next_value = self.get_value(next_obs, latent=latent)
        value = self.get_value(obs, latent=latent)
        log_p_tau = reward + gamma * next_value - value
        log_p_tau = log_p_tau.clamp(-20, 10).reshape(-1)
        log_q_tau = log_q_tau.clamp(-20, 10).reshape(-1)
        sum_and_exp = torch.exp(log_p_tau) + torch.exp(log_q_tau)
        log_pq = torch.log(sum_and_exp.clamp(1e-6, 10e6))
        return log_p_tau, log_q_tau, log_pq, reward, value, next_value

    def compute_discriminator_loss(self, *, obs, act, next_obs, action_logp, latent, labels, gamma, expert_traj, carsize):
        stats = {}

        if self.algorithm in ["airl"]:
            log_p_tau, log_q_tau, log_pq, reward, value, _ = self._get_airl_logits(
                obs=obs, act=act, next_obs=next_obs, action_logp=action_logp, gamma=gamma,
                latent=latent
            )
            loss = labels * (log_p_tau - log_pq) + (1 - labels) * (log_q_tau - log_pq)
            # loss = loss[valid_mask]
            loss = -torch.mean(loss)
            # Note: L2 loss is replaced by the weight decay in Adam optimizer.
            # self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params]) * self.l2_loss_ratio
            with torch.no_grad():
                scores = torch.exp(log_p_tau - log_pq)
                # Pred -> 1, should be expert sample, should labels=1.
                acc = torch.sum(((scores > 0.5) == labels)) / obs.shape[0]
                stats.update({
                    "discriminator_value_max": value.max().item(),
                    "discriminator_value_mean": value.mean().item(),
                    "discriminator_value_min": value.min().item(),
                    "discriminator_reward_max": reward.max().item(),
                    "discriminator_reward_mean": reward.mean().item(),
                    "discriminator_reward_min": reward.min().item(),
                    "discriminator_accuracy": acc.item()
                })

        elif self.algorithm in ["gail"]:
            # TODO: add a flag to support raw obs and raw state
            reward = self.get_reward(obs=obs, next_obs=next_obs)
            loss = self._gail_discriminator_loss(input=reward, target=labels)

            # discriminator accuracy
            with torch.no_grad():
                # Ret -> 1, should be human sample, should labels=1.
                acc = torch.sum(((reward > 0.5) == labels)) / reward.shape[0]
                stats.update({
                    "discriminator_reward_max": reward.max().item(),
                    "discriminator_reward_mean": reward.mean().item(),
                    "discriminator_reward_min": reward.min().item(),
                    "discriminator_accuracy": acc.item()
                })

        else:
            raise ValueError()

        return loss, stats

    # def get_next_state(self, state, action, dynamics_parameters, latent=None, add_state=True):
    def get_next_state(self, timesteps, state, action, dynamics, mask):
        """
        This is the old implementation of next state.
        In transformer-based forward model, we should input the timesteps and so on to the model.
        """
        # state in [0, 1]
        # action in [-1, 1], should be rescaled to [0, 1]
        # dynamics_parameters in [-1, 1]
        # if state.shape[1] == 270:  # Hardcoded!
        #     state = state[:, :-1]
        # action = (action.clamp(-1, 1) + 1) / 2
        # dynamics_parameters = (dynamics_parameters.clamp(-1, 1) + 1) / 2
        # inp = torch.cat([state, action, dynamics_parameters], dim=-1)
        # # if self.enable_latent:
        # #     inp = torch.cat([latent, inp], dim=-1)
        # diff = self._forward_model(inp)
        # if add_state:
        #     assert state.shape[1] == 269 + self.latent_dim
        #     ret = state[:, self.latent_dim:] + diff
        # else:
        #     ret = diff

        # Transformer-based forward model calling
        ret = self._forward_model(timesteps, state, action, dynamics, mask)

        return ret

    def compute_forward_loss(self, *, state, next_state, action, dynamics_parameters, latent):
        """
        This is the old function to compute forward loss, when we use a MLP as the forward model.
        Now we are using transformer-based forward model, thus the input requires more processing.
        """
        raise ValueError()
        valid_mask = dynamics_parameters.mean(dim=1) != -10.0

        if not torch.any(valid_mask):
            return None, None

        pred_next_state_diff = self.get_next_state(state=state, action=action, dynamics_parameters=dynamics_parameters,
                                              latent=latent, add_state=False)

        pred_next_state_diff = pred_next_state_diff[valid_mask]

        if next_state.shape[1] == 270:  # Hardcoded!
            next_state = next_state[:, :-1]
            state = state[:, :-1]
        if self.enable_latent:
            next_state = next_state[:, self.latent_dim:]
            state = state[:, self.latent_dim:]
        target = next_state[valid_mask] - state[valid_mask]

        forward_loss = self._forward_model_loss(input=pred_next_state_diff, target=target)

        stat = dict(
            forward_pred_max=pred_next_state_diff.max().item(),
            forward_pred_min=pred_next_state_diff.min().item(),
            forward_pred_mean=pred_next_state_diff.mean().item(),
            forward_real_mean=target.mean().item(),
        )
        return forward_loss, stat

    def compute_dynamics_loss(self, *, state, action, action_logp, latent, gamma, device=None):
        # TODO: add a flag to support raw obs and raw state

        # If using dynamics_policy, do not enter here!
        assert self.randomized_dynamics != DYNAMICS_POLICY


        dynamics_posterior_weight = self.model_config["custom_model_config"]["dynamics_posterior_weight"]
        dynamics_entropy = None

        # Sample from current distribution
        if self.randomized_dynamics == "std":
            # Important! Sample from
            d_mean, d_log_std = self.get_dynamics_parameters_distribution(device=device)
            d_std = torch.exp(d_log_std.clamp(-20, 10))
            dist = Normal(d_mean, d_std)
            dynamics_parameters = dist.rsample([state.shape[0]])

        elif self.randomized_dynamics == "gmm":
            dynamics_matrix, weights = self.get_dynamics_parameters_distribution(
                nn_batch_size=state.shape[0], latent=state[:, :self.latent_dim], device=device
            )
            # dynamics_matrix is in [K, dynamics dim * 2]
            # weights is in [bs, K]

            one_hot = gumbel_softmax(logits=weights, temperature=1.0, device=device)
            # one_hot is in [bs, K]

            selected_dynamics = one_hot @ dynamics_matrix
            # in [bs, dynamics dim * 2]

            loc, loc_std = torch.chunk(selected_dynamics, 2, dim=-1)
            d_std = torch.exp(loc_std.clamp(-20, 10))
            dist = Normal(loc, d_std)
            dynamics_parameters = dist.rsample()
            # dynamics_parameters is in [bs, dynamics dim]

            p_log_p = weights * torch.nn.functional.softmax(weights, dim=1)
            dynamics_entropy = torch.mean(-p_log_p.sum(-1))

        elif self.randomized_dynamics == "mean":
            dynamics_parameters = self.dynamics_parameters

            # Sample a lots of dynamics_parameters to fit the shape
            dynamics_parameters = dynamics_parameters.repeat(state.shape[0], 1)

        elif self.randomized_dynamics in ["nn", "nnstd"]:

            # Note: Here we randomly sample a new input (noise) to the dynamics model.
            # One option is to use the old noise that we used to sample the dynamics that was applied to the env.
            # But I don't see much difference here. And using current impl allows multiple SGD epochs.
            dynamics_parameters = self.get_dynamics_parameters_distribution(
                nn_batch_size=state.shape[0], latent=latent, device=device
            )

        else:
            raise ValueError()

        dynamics_parameters = dynamics_parameters.clamp(-1, 1)
        raw_dynamics_parameters = dynamics_parameters.cpu().detach().numpy()
        # in [-1, 1]

        # clip action? Correct, already clipped.
        # Call the forward model to get the next state.
        fake_next_obs = self.get_next_state(state, action, dynamics_parameters, latent=latent)

        # clip fake next obs? We should!
        fake_next_obs = fake_next_obs.clamp(0, 1)

        if self.enable_latent:
            fake_next_obs = torch.cat([state[:, :self.latent_dim], fake_next_obs], dim=1)

        if state.shape[1] == 270:
            state = state[:, :-1]

        if self.algorithm in ["airl"]:
            log_p_tau, log_q_tau, log_pq, reward, value, _ = self._get_airl_logits(
                obs=state, act=action, next_obs=fake_next_obs, action_logp=action_logp, gamma=gamma,
                latent=latent
            )
            scores = torch.exp(log_p_tau - log_pq)

        elif self.algorithm in ["gail"]:
            scores = self.get_reward(obs=state, next_obs=fake_next_obs)
            # GAIL scores is after sigmoid, so it's in [0, 1]
        else:
            raise ValueError()

        # scores should be in [0, 1]
        scores = torch.clamp(scores, 1e-6, 1 - 1e-6)
        labels = torch.ones([action.shape[0]]).to(action.device)
        loss = nn.functional.binary_cross_entropy(input=scores, target=labels)

        stat = dict(
            dynamics_loss=loss.item(),
            dynamics_scores=scores.mean().item(),
            dynamics_mean=np.mean(raw_dynamics_parameters),
            dynamics_max=np.max(raw_dynamics_parameters),
            dynamics_min=np.min(raw_dynamics_parameters),
            dynamics_std=np.std(raw_dynamics_parameters),
        )

        if dynamics_posterior_weight > 0 and self.enable_latent:
            pred_latent = self._dynamics_model_reverse(dynamics_parameters)
            loc, loc_std = torch.chunk(pred_latent, 2, dim=-1)
            d_std = torch.exp(loc_std.clamp(-20, 10))
            dist = Normal(loc, d_std)
            logp = dist.log_prob(state[:, :self.latent_dim])
            reverse_loss = -logp.mean()
            loss += dynamics_posterior_weight * reverse_loss
            stat["dynamics_reverse_loss"] = reverse_loss.item()

        if dynamics_entropy is not None:
            stat["dynamics_entropy"] = dynamics_entropy.item()
        return loss, stat

    def compute_latent_dict(self, expert_traj_dict, max_trajectory_len, device):
        batch = dict(expert_state=[], expert_timestep=[], expert_mask=[])
        for unique, traj in expert_traj_dict.items():
            # Every "episode" is a trajectory of one agent
            # traj = episode["obs"]
            """
            [0, 10) - latent
            [10, 130) - side detector
            [130, 148) - state info
            [148, 158) - navi info
            [158, 159) - lateral dist
            [159, 279) - lidar
            [279, 280) - lcf

            total - 280
            """

            if self.randomized_dynamics == DYNAMICS_POLICY:
                pass

            else:
                if traj.shape[1] == 279:
                    traj = traj[:, 130:159]
                elif traj.shape[1] == 280:
                    traj = traj[:, 130:159]
                elif traj.shape[1] == 269:
                    traj = traj[:, 120:149]
            # else:
            #     raise ValueError()

            expert_state, expert_timestep, expert_mask = _pad_a_traj(traj, max_trajectory_len, should_slice=self.randomized_dynamics != DYNAMICS_POLICY)
            batch["expert_state"].append(expert_state)
            batch["expert_timestep"].append(expert_timestep)
            batch["expert_mask"].append(expert_mask)
        batch = {k: convert_to_torch_tensor(torch.stack(v), device=device) for k, v in batch.items()}
        batch = SampleBatch(batch)
        ret = self.get_latent(batch["expert_timestep"], batch["expert_state"], batch["expert_mask"])
        latent = convert_to_numpy(ret)

        latent_dict = {}
        for count, k in enumerate(expert_traj_dict.keys()):
            latent_dict[k] = latent[count]

        return latent_dict

# ModelCatalog.register_custom_model("discriminator", Discriminator)


# ========== New Policy ==========
def compute_nei_advantage(rollout: SampleBatch, last_r: float, gamma: float = 0.9, lambda_: float = 1.0):
    vpred_t = np.concatenate([rollout[NEI_VALUES], np.array([last_r])])
    delta_t = (rollout[NEI_REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
    rollout[NEI_ADVANTAGE] = discount_cumsum(delta_t, gamma * lambda_)
    rollout[NEI_TARGET] = (rollout[NEI_ADVANTAGE] + rollout[NEI_VALUES]).astype(np.float32)
    rollout[NEI_ADVANTAGE] = rollout[NEI_ADVANTAGE].astype(np.float32)
    return rollout


def compute_global_advantage(rollout: SampleBatch, last_r: float, gamma: float = 1.0, lambda_: float = 1.0):
    vpred_t = np.concatenate([rollout[GLOBAL_VALUES], np.array([last_r])])
    delta_t = (rollout[GLOBAL_REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
    rollout[GLOBAL_ADVANTAGES] = discount_cumsum(delta_t, gamma * lambda_)
    rollout[GLOBAL_TARGET] = (rollout[GLOBAL_ADVANTAGES] + rollout[GLOBAL_VALUES]).astype(np.float32)
    rollout[GLOBAL_ADVANTAGES] = rollout[GLOBAL_ADVANTAGES].astype(np.float32)
    return rollout


class NewPolicy(CCPPOPolicy):
    def __init__(self, observation_space, action_space, config):
        # Compatibility for different gym version
        if isinstance(observation_space, Box) and not hasattr(observation_space, "_shape"):
            observation_space._shape = observation_space.shape
        if isinstance(action_space, Box) and not hasattr(action_space, "_shape"):
            action_space._shape = action_space.shape

        super(NewPolicy, self).__init__(observation_space, action_space, config)

        # Add a target model
        dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"], framework="torch")
        self.target_model = ModelCatalog.get_model_v2(
            obs_space=observation_space,
            action_space=action_space,
            num_outputs=logit_dim,
            model_config=config["model"],
            framework="torch",
            name="new_target_model"
        )
        self.target_model.to(self.device)
        self.update_old_policy()  # Note that we are not sure if the model is synced with local policy at this time

        # ========== Modification! ==========
        config["model"]["discriminator_add_action"] = config["discriminator_add_action"]
        config["model"]["discriminator_use_tanh"] = config["discriminator_use_tanh"]

        assert not self.config[COUNTERFACTUAL]  # This is a config of CoPO
        self.discriminator = Discriminator(
            obs_space=observation_space,
            action_space=action_space,
            num_outputs=logit_dim,
            model_config=config["model"],
            name="discriminator",
            # obs_dim=269  # Hardcoded here! Tired to pass many arguments here and there
        )

        self.discriminator.to(self.device)

        if self.config["enable_latent_posterior"]:
            self._posterior_optimizer = torch.optim.Adam(
                self.discriminator.get_posterior_parameters(),
                lr=self.config["posterior_lr"],
                weight_decay=self.config["discriminator_l2"]
            )
            self._discriminator_optimizer = torch.optim.Adam(
                self.discriminator.get_discriminator_parameters(),
                lr=self.config["discriminator_lr"],
                weight_decay=self.config["discriminator_l2"]  # 0.1 in MA-AIRL impl.
            )
        else:
            self._discriminator_optimizer = torch.optim.Adam(
                self.discriminator.get_discriminator_parameters(),
                lr=self.config["discriminator_lr"],
                weight_decay=self.config["discriminator_l2"]  # 0.1 in MA-AIRL impl.
            )

        if self.config["algorithm"] in ["airl"]:
            self._inverse_optimizer = torch.optim.Adam(
                self.discriminator.get_inverse_parameters(),
                lr=self.config["inverse_lr"],  # Just reuse the hyperparameters!!!
                weight_decay=self.config["discriminator_l2"]  # Just reuse the hyperparameters!!!
            )

        if self.config["randomized_dynamics"]:
            self._forward_optimizer = torch.optim.Adam(
                self.discriminator.get_forward_parameters(),
                lr=self.config["inverse_lr"],  # Just reuse the hyperparameters!!!
                weight_decay=self.config["discriminator_l2"]  # Just reuse the hyperparameters!!!
            )
            # if self.config["enable_latent_posterior"]:
            #     dy_model_para = self.discriminator.get_dynamics_parameters() + self.discriminator.get_posterior_parameters()
            # else:
            dy_model_para = self.discriminator.get_dynamics_parameters()
            if self.config["dynamics_posterior_weight"] > 0:
                dy_model_para += self.discriminator.get_posterior_parameters()
            self._dynamics_optimizer = torch.optim.Adam(
                dy_model_para,
                lr=self.config["dynamics_lr"],  # Just reuse the hyperparameters!!!
                weight_decay=self.config["discriminator_l2"]  # Just reuse the hyperparameters!!!
            )

        # Setup the LCF optimizer and relevant variables.
        # Note that this optimizer is only used in local policy, but it is defined in all policies.
        self._lcf_optimizer = torch.optim.Adam([self.model.lcf_parameters], lr=self.config[LCF_LR])

        self.view_requirements[SampleBatch.ACTIONS] = ViewRequirement(space=action_space)
        self.view_requirements[SampleBatch.NEXT_OBS] = ViewRequirement(
            SampleBatch.OBS,
            space=observation_space,
            shift=1
        )

    def update_inverse_model(self, train_batch):
        # train_batch = self._lazy_tensor_dict(train_batch, device=self.device)

        # Build a mixed batch first!

        # processed_batch = train_batch

        if isinstance(self.action_space, Box):
            clipped_actions = np.clip(train_batch["actions"], self.action_space.low[0], self.action_space.high[0])
        else:
            clipped_actions = train_batch["actions"].astype(int)

        inverse_loss, pseudo_action = self.discriminator.compute_inverse_loss(
            obs=convert_to_torch_tensor(train_batch[SampleBatch.OBS], device=self.device),
            next_obs=convert_to_torch_tensor(train_batch[SampleBatch.NEXT_OBS], device=self.device),
            actions=convert_to_torch_tensor(clipped_actions, device=self.device),
            low=self.action_space.low[0] if isinstance(self.action_space, Box) else None,
            high=self.action_space.high[0] if isinstance(self.action_space, Box) else None,
            latent=None,
            # latent=convert_to_torch_tensor(train_batch["latent"], device=self.device) \
            #     if self.config["enable_latent"] else None,
            # dynamics_parameters=convert_to_torch_tensor(train_batch["dynamics_parameters"], device=self.device),
            dynamics_parameters=None,
        )
        self._inverse_optimizer.zero_grad()
        inverse_loss.backward()
        self._inverse_optimizer.step()

        if isinstance(self.action_space, Box):
            l1_diff = convert_to_numpy(pseudo_action) - clipped_actions
            stats = {
                "inverse_loss": inverse_loss.item(),
                "inverse_abs_mean": np.mean(np.abs(l1_diff)),
                "inverse_abs_dim0_mean": np.mean(np.abs(l1_diff)[:, 0]),
                "inverse_abs_dim1_mean": np.mean(np.abs(l1_diff)[:, 1]),
                "inverse_l1_diff_max": np.max(l1_diff),
                "inverse_l1_diff_min": np.min(l1_diff),
                "inverse_l1_diff_mean": np.mean(l1_diff),
                "inverse_pseudo_action_mean": pseudo_action.mean().item(),
                "inverse_pseudo_action_min": pseudo_action.min().item(),
                "inverse_pseudo_action_max": pseudo_action.max().item(),
            }

        else:
            acc = (convert_to_numpy(pseudo_action.max(1)[1]) == clipped_actions).sum() / clipped_actions.shape[0]
            # acc = (clipped_actions == pseudo_action.max(1)).sum() / clipped_actions.size(0)
            stats = {
                "inverse_loss": inverse_loss.item(),
                "inverse_accuracy": acc.item(),
                # "inverse_abs_mean": np.mean(np.abs(l1_diff)),
                # "inverse_abs_dim0_mean": np.mean(np.abs(l1_diff)[:, 0]),
                # "inverse_abs_dim1_mean": np.mean(np.abs(l1_diff)[:, 1]),
                # "inverse_l1_diff_max": np.max(l1_diff),
                # "inverse_l1_diff_min": np.min(l1_diff),
                # "inverse_l1_diff_mean": np.mean(l1_diff),
                # "inverse_pseudo_action_mean": pseudo_action.mean().item(),
                # "inverse_pseudo_action_min": pseudo_action.min().item(),
                # "inverse_pseudo_action_max": pseudo_action.max().item(),
            }

        return stats

    def update_discriminator(self, train_batch):
        # train_batch = self._lazy_tensor_dict(train_batch, device=self.device)

        if self.config["algorithm"] in ["airl"]:
            if isinstance(self.action_space, Box):
                clipped_actions = np.clip(train_batch["actions"], self.action_space.low[0], self.action_space.high[0])
            else:
                clipped_actions = action_discrete_to_continuous(
                    train_batch["actions"], discrete_action_dim=self.config["discrete_action_dim"]
                )
            action = convert_to_torch_tensor(clipped_actions, device=self.device)
            action_logp = convert_to_torch_tensor(train_batch[SampleBatch.ACTION_LOGP], device=self.device)
        else:
            action = None
            action_logp = None
        discriminator_loss, stats = self.discriminator.compute_discriminator_loss(
            obs=convert_to_torch_tensor(train_batch[SampleBatch.OBS], device=self.device),
            act=action,
            next_obs=convert_to_torch_tensor(train_batch[SampleBatch.NEXT_OBS], device=self.device),
            action_logp=action_logp,
            labels=convert_to_torch_tensor(train_batch["labels"], device=self.device),
            gamma=self.config["gamma"],
            # add_action=self.config["discriminator_add_action"],
            # latent=convert_to_torch_tensor(train_batch["latent"], device=self.device) \
            #     if self.config["enable_latent"] else None,
            latent=None,
            expert_traj=None,
            # expert_traj=convert_to_torch_tensor(train_batch["expert_traj"], device=self.device) \
            #     if self.config["enable_latent_posterior"] else None,
            carsize=convert_to_torch_tensor(train_batch["carsize"], device=self.device)
        )
        self._discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self._discriminator_optimizer.step()

        return stats

    def update_forward_model(self, train_batch):
        pred_state, action, dynamics = self.discriminator.get_next_state(
            timesteps=train_batch["timesteps"],
            state=train_batch["states"],
            action=train_batch["actions"],
            dynamics=train_batch["dynamics"],
            mask=train_batch["traj_mask"]
        )

        forward_loss = nn.functional.mse_loss(input=pred_state, target=train_batch["next_states"])

        stat = {}
        self._forward_optimizer.zero_grad()
        forward_loss.backward()
        self._forward_optimizer.step()
        stat["forward_loss"] = forward_loss.item()

        return stat


        # === PZH: The following script are deprecated they are for MLP forward model ===
        pred_next_state_diff = pred_next_state_diff[valid_mask]

        if next_state.shape[1] == 270:  # Hardcoded!
            next_state = next_state[:, :-1]
            state = state[:, :-1]
        if self.enable_latent:
            next_state = next_state[:, self.latent_dim:]
            state = state[:, self.latent_dim:]
        target = next_state[valid_mask] - state[valid_mask]

        forward_loss = self._forward_model_loss(input=pred_next_state_diff, target=target)

        stat = dict(
            forward_pred_max=pred_next_state_diff.max().item(),
            forward_pred_min=pred_next_state_diff.min().item(),
            forward_pred_mean=pred_next_state_diff.mean().item(),
            forward_real_mean=target.mean().item(),
        )




        forward_loss, stat = self.discriminator.compute_forward_loss(
            state=convert_to_torch_tensor(train_batch[SampleBatch.OBS], device=self.device),
            next_state=convert_to_torch_tensor(train_batch[SampleBatch.NEXT_OBS], device=self.device),
            action=convert_to_torch_tensor(train_batch["actions"], device=self.device),
            dynamics_parameters=convert_to_torch_tensor(train_batch["dynamics_parameters"], device=self.device),
            latent=None,
            # latent=convert_to_torch_tensor(train_batch["latent"], device=self.device) \
            #     if self.config["enable_latent"] else None,
        )







        if forward_loss is not None:

            self._forward_optimizer.zero_grad()
            forward_loss.backward()
            self._forward_optimizer.step()

            stat["forward_loss"] = forward_loss.item()

        else:
            stat = {}

        return stat

    def update_dynamics(self, train_batch):
        if isinstance(self.action_space, Box):
            clipped_actions = np.clip(train_batch["actions"], self.action_space.low[0], self.action_space.high[0])
        else:
            clipped_actions = action_discrete_to_continuous(train_batch["actions"],
                                                            discrete_action_dim=self.config["discrete_action_dim"])
        dynamics_loss, stats = self.discriminator.compute_dynamics_loss(
            state=convert_to_torch_tensor(train_batch[SampleBatch.OBS], device=self.device),
            # next_state=convert_to_torch_tensor(train_batch[SampleBatch.NEXT_OBS], device=self.device),
            action=convert_to_torch_tensor(clipped_actions, device=self.device),
            # dynamics_parameters=convert_to_torch_tensor(train_batch["dynamics_parameters"], device=self.device)
            action_logp=convert_to_torch_tensor(train_batch[SampleBatch.ACTION_LOGP], device=self.device),
            # add_action=self.config["discriminator_add_action"],
            gamma=self.config["gamma"],
            device=self.device,
            latent=None,
            # latent=convert_to_torch_tensor(train_batch["latent"], device=self.device) \
            #     if self.config["enable_latent"] else None,
        )

        self._dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        self._dynamics_optimizer.step()
        return stats

    def update_dynamics_policy(self, train_batch):
        dynamics_logit = self.discriminator._dynamics_model(train_batch["latent"])
        loc, log_std = torch.chunk(dynamics_logit, 2, dim=1)
        std = torch.exp(log_std.clamp(-20, 10))
        dist = Normal(loc, std)

        logp_ratio = torch.exp(
            dist.log_prob(train_batch["dynamics_parameters"]).sum(-1)
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        surrogate_loss = torch.min(
            train_batch["reward"] * logp_ratio,
            train_batch["reward"] * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )
        loss = -surrogate_loss.mean()


        # loss = dist.log_prob(train_batch["dynamics_parameters"]).sum(-1) * train_batch["reward"]
        # loss = -loss.mean()

        stat = dict(
            dynamics_loss=loss.item(),
            dynamics_scores=train_batch["reward"].mean().item(),
            dynamics_scores_max=train_batch["reward"].max().item(),
            dynamics_scores_min=train_batch["reward"].min().item(),
            dynamics_mean=torch.mean(train_batch["dynamics_parameters"]).item(),
            dynamics_max=torch.max(train_batch["dynamics_parameters"]).item(),
            dynamics_min=torch.min(train_batch["dynamics_parameters"]).item(),
            dynamics_std=torch.std(train_batch["dynamics_parameters"]).item(),
        )
        self._dynamics_optimizer.zero_grad()
        loss.backward()
        self._dynamics_optimizer.step()
        return stat

    def update_dynamics_policy_old(self, train_batch):
        """
        Use RL method to update dynamics policy

        The policy itself takes latent code of a trajectory as input
        It outputs 5 dim dynamics parameter for the trajectory.

        Given the initial state, latent code and the new dynamics of the trajectory,
        we can evaluate the "improvement" brought by the dynamics change.
        """
        dynamics_entropy = None

        # We can call the environment to evaluate?
        if isinstance(self.action_space, Box):
            clipped_actions = np.clip(train_batch["actions"], self.action_space.low[0], self.action_space.high[0])
        else:
            clipped_actions = action_discrete_to_continuous(train_batch["actions"],
                                                            discrete_action_dim=self.config["discrete_action_dim"])
        state=convert_to_torch_tensor(train_batch[SampleBatch.OBS], device=self.device)
        action=convert_to_torch_tensor(clipped_actions, device=self.device)
        action_logp=convert_to_torch_tensor(train_batch[SampleBatch.ACTION_LOGP], device=self.device)
        gamma=self.config["gamma"]
        device=self.device
        latent = None

        dynamics_parameters = self.discriminator.get_dynamics_parameters_distribution(
            nn_batch_size=state.shape[0], latent=state[:, :self.discriminator.latent_dim], device=device
        )

        # dynamics_logit = self.discriminator._dynamics_model(state[:, :self.discriminator.latent_dim])
        # loc, log_std = torch.chunk(dynamics_logit, 2, dim=1)
        # std = torch.exp(log_std.clamp(-20, 10))
        # dist = Normal(loc, std)
        # dynamics_parameters = dist.sample()

        dynamics_parameters = dynamics_parameters.clamp(-1, 1)
        raw_dynamics_parameters = dynamics_parameters.cpu().detach().numpy()
        # in [-1, 1]

        # clip action? Correct, already clipped.
        # Call the forward model to get the next state.
        fake_next_obs = self.discriminator.get_next_state(state, action, dynamics_parameters, latent=None)

        # clip fake next obs? We should!
        fake_next_obs = fake_next_obs.clamp(0, 1)

        if self.config["enable_latent"]:
            fake_next_obs = torch.cat([state[:, :self.discriminator.latent_dim], fake_next_obs], dim=1)

        if state.shape[1] == 270:
            state = state[:, :-1]

        if self.config["algorithm"] in ["airl"]:
            log_p_tau, log_q_tau, log_pq, reward, value, _ = self.discriminator._get_airl_logits(
                obs=state, act=action, next_obs=fake_next_obs, action_logp=action_logp, gamma=gamma,
                latent=latent
            )
            scores = torch.exp(log_p_tau - log_pq)

        elif self.config["algorithm"] in ["gail"]:
            scores = self.discriminator.get_reward(obs=state, next_obs=fake_next_obs)
            # GAIL scores is after sigmoid, so it's in [0, 1]
        else:
            raise ValueError()

        # scores should be in [0, 1]
        scores = torch.clamp(scores, 1e-6, 1 - 1e-6)
        labels = torch.ones([action.shape[0]]).to(action.device)
        loss = nn.functional.binary_cross_entropy(input=scores, target=labels)

        # REINFORCE loss:
        # loss = dist.log_prob(dynamics_parameters) * scores
        # loss = -loss.mean()

        stat = dict(
            dynamics_loss=loss.item(),
            dynamics_scores=scores.mean().item(),
            dynamics_mean=np.mean(raw_dynamics_parameters),
            dynamics_max=np.max(raw_dynamics_parameters),
            dynamics_min=np.min(raw_dynamics_parameters),
            dynamics_std=np.std(raw_dynamics_parameters),
        )

        # if dynamics_posterior_weight > 0 and self.config["enable_latent"]:
        #     pred_latent = self.discriminator._dynamics_model_reverse(dynamics_parameters)
        #     loc, loc_std = torch.chunk(pred_latent, 2, dim=-1)
        #     d_std = torch.exp(loc_std.clamp(-20, 10))
        #     dist = Normal(loc, d_std)
        #     logp = dist.log_prob(state[:, :self.discriminator.latent_dim])
        #     reverse_loss = -logp.mean()
        #     loss += dynamics_posterior_weight * reverse_loss
        #     stat["dynamics_reverse_loss"] = reverse_loss.item()

        if dynamics_entropy is not None:
            stat["dynamics_entropy"] = dynamics_entropy.item()

        self._dynamics_optimizer.zero_grad()
        loss.backward()
        self._dynamics_optimizer.step()

        return stat

    def update_posterior(self, train_batch, contrastive_batch, posterior_match_loss_weight, num_neg_samples=1):
        # We should change this future.
        posterior_loss_list = []
        posterior_prob_list = []
        for batch in minibatches(contrastive_batch, 512):
            origin = self.discriminator.get_latent(
                batch["origin_timestep"],
                batch["origin_state"],
                batch["origin_mask"],
                # use_target=True
            )

            pos = self.discriminator.get_latent(
                batch["pos_timestep"],
                batch["pos_state"],
                batch["pos_mask"],
                use_target=True
            ).detach()
            pos_pair_logit = torch.einsum('nc,nc->n', [origin, pos])

            neg_latent_list = []
            for x in range(num_neg_samples):
                neg = self.discriminator.get_latent(
                    batch["neg_timestep_{}".format(x)],
                    batch["neg_state_{}".format(x)],
                    batch["neg_mask_{}".format(x)],
                    use_target=True
                ).detach()
                neg_pair_logit = torch.einsum('nc,nc->n', [origin, neg])
                neg_latent_list.append(neg_pair_logit)

            logit = torch.stack([pos_pair_logit] + neg_latent_list, dim=-1)

            # Temperature from MOCO paper
            logit /= self.config["posterior_temperature"]

            log_prob = torch.nn.functional.log_softmax(logit, dim=-1)[:, 0]
            posterior_loss = -torch.mean(log_prob)

            self._posterior_optimizer.zero_grad()
            posterior_loss.backward()
            self._posterior_optimizer.step()

            posterior_loss_list.append(posterior_loss.item())
            posterior_prob_list.append(torch.exp(log_prob).mean().item())

            self.discriminator.update_posterior_target()

        expert_agent_match_loss_list = []
        for batch in minibatches(train_batch, 512):
            expert_latent = self.discriminator.get_latent(
                batch["expert_timestep"],
                batch["expert_state"],
                batch["expert_mask"],
                use_target=True
            ).detach()

            agent_latent = self.discriminator.get_latent(
                batch["agent_timestep"],
                batch["agent_state"],
                batch["agent_mask"],
                # use_target=True
            )

            # Loss to match agent latent with expert latent:
            expert_agent_match_loss = nn.functional.mse_loss(input=agent_latent, target=expert_latent)

            self._posterior_optimizer.zero_grad()
            (expert_agent_match_loss * posterior_match_loss_weight).backward()
            self._posterior_optimizer.step()

            expert_agent_match_loss_list.append(expert_agent_match_loss.item())

            self.discriminator.update_posterior_target()

        return {
            "posterior_contrastive_loss": np.mean(posterior_loss_list),
            # "posterior_similarity1": sim1.item(),
            # "posterior_similarity2": sim2.item(),
            "posterior_prob": np.mean(posterior_prob_list),
            "posterior_match_loss": np.mean(expert_agent_match_loss_list)
        }

    def meta_update(self, train_batch):
        # Build the loss between new policy and ego advantage.
        train_batch = self._lazy_tensor_dict(train_batch, device=self.device)
        logits, state = self.model(train_batch)
        curr_action_dist = self.dist_class(logits, self.model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=self.model.is_time_major())
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
            train_batch[SampleBatch.ACTION_LOGP]
        )

        adv = train_batch[GLOBAL_ADVANTAGES]
        surrogate_loss = torch.min(
            adv * logp_ratio,
            adv * torch.clamp(logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"])
        )
        new_policy_loss = reduce_mean_valid(-surrogate_loss)
        self.model.zero_grad()
        new_policy_grad = torch.autograd.grad(new_policy_loss, self.model.parameters(), allow_unused=True)
        new_policy_grad = [g for g in new_policy_grad if g is not None]

        # Build the loss between old policy and old log prob.
        old_logits, old_state = self.target_model(train_batch)
        old_dist = self.dist_class(old_logits, self.target_model)
        old_logp = old_dist.logp(train_batch[SampleBatch.ACTIONS])
        assert old_logp.ndim == 1
        old_policy_loss = reduce_mean_valid(old_logp)
        self.target_model.zero_grad()
        old_policy_grad = torch.autograd.grad(old_policy_loss, self.target_model.parameters(), allow_unused=True)
        old_policy_grad = [g for g in old_policy_grad if g is not None]

        grad_value = 0
        assert len(new_policy_grad) == len(old_policy_grad)
        for a, b in zip(new_policy_grad, old_policy_grad):
            assert a.shape == b.shape
            grad_value += (a * b).sum()

        # Build the loss between LCF and LCF advantage
        # Note: the adv_mean/std is the training batch-averaged mean/std of LCF advantage. But here we are in a
        #  minibatch.
        advantages = self.model.compute_coordinated(
            ego=train_batch[Postprocessing.ADVANTAGES], neighbor=train_batch[NEI_ADVANTAGE]
        )
        lcf_advantages = (advantages - self._raw_lcf_adv_mean) / self._raw_lcf_adv_std
        lcf_lcf_adv_loss = reduce_mean_valid(lcf_advantages)
        lcf_final_loss = grad_value * lcf_lcf_adv_loss
        self._lcf_optimizer.zero_grad()
        lcf_final_loss.backward()
        self._lcf_optimizer.step()

        stats = dict(
            new_policy_ego_loss=new_policy_loss,
            old_policy_logp_loss=old_policy_loss,
            lcf_lcf_adv_loss=lcf_lcf_adv_loss,
            lcf_final_loss=lcf_final_loss.item(),
            grad_value=grad_value.item(),
            lcf=self.model.lcf_mean.item(),
            lcf_deg=self.model.lcf_mean.item() * 90,
            lcf_param=self.model.lcf_parameters[0].item(),
            coordinated_adv=reduce_mean_valid(advantages),
            global_adv=reduce_mean_valid(adv),
        )
        if self.config[USE_DISTRIBUTIONAL_LCF]:
            stats.update(lcf_std=self.model.lcf_std.item())
            stats.update(lcf_std_deg=self.model.lcf_std.item() * 90)
            stats.update(lcf_std_param=self.model.lcf_parameters[1].item())
        return stats

    def loss(self, model, dist_class, train_batch):
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major())
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        # Modification: The advantage is changed here!
        # Note: the advantages here is already modified by the coordinated advantage!
        advantages = train_batch["normalized_advantages"]

        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            )
        )
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        # Compute a value function loss.
        assert self.config["use_critic"]

        if self.config["old_value_loss"]:
            def _compute_value_loss(current_vf, prev_vf, value_target):
                vf_loss1 = torch.pow(current_vf - value_target, 2.0)
                vf_clipped = prev_vf + torch.clamp(
                    current_vf - prev_vf, -self.config["vf_clip_param"], self.config["vf_clip_param"]
                )
                vf_loss2 = torch.pow(vf_clipped - value_target, 2.0)
                vf_loss = torch.max(vf_loss1, vf_loss2)
                return vf_loss

        else:

            def _compute_value_loss(current_vf, prev_vf, value_target):
                vf_loss = torch.pow(
                    current_vf - value_target, 2.0
                )
                vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
                return vf_loss_clipped

        native_vf_loss = _compute_value_loss(
            current_vf=model.central_value_function(train_batch[CENTRALIZED_CRITIC_OBS]),
            prev_vf=train_batch[SampleBatch.VF_PREDS],
            value_target=train_batch[Postprocessing.VALUE_TARGETS]
        )
        mean_native_vf_loss = reduce_mean_valid(native_vf_loss)

        if self.config["enable_copo"]:
            nei_vf_loss = _compute_value_loss(
                current_vf=model.get_nei_value(train_batch[CENTRALIZED_CRITIC_OBS]),
                prev_vf=train_batch[NEI_VALUES],
                value_target=train_batch[NEI_TARGET]
            )
            nei_mean_vf_loss = reduce_mean_valid(nei_vf_loss)

            global_vf_loss = _compute_value_loss(
                current_vf=model.get_global_value(train_batch[CENTRALIZED_CRITIC_OBS]),
                prev_vf=train_batch[GLOBAL_VALUES],
                value_target=train_batch[GLOBAL_TARGET]
            )
            global_mean_vf_loss = reduce_mean_valid(global_vf_loss)

        else:
            nei_vf_loss = 0.0
            global_vf_loss = 0.0

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * native_vf_loss
            + self.config["vf_loss_coeff"] * nei_vf_loss
            + self.config["vf_loss_coeff"] * global_vf_loss
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_vf_loss"] = mean_native_vf_loss
        model.tower_stats["vf_explained_var"] = torch.zeros(())
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        model.tower_stats["normalized_advantages"] = advantages.mean()

        # Modification: newly add stats
        if self.config["enable_copo"]:
            model.tower_stats["lcf"] = model.lcf_mean
            if self.config[USE_DISTRIBUTIONAL_LCF]:
                model.tower_stats["lcf_std"] = model.lcf_std
            model.tower_stats["mean_nei_vf_loss"] = nei_mean_vf_loss
            model.tower_stats["mean_global_vf_loss"] = global_mean_vf_loss

        return total_loss

    def stats_fn(self, train_batch):
        ret = super(NewPolicy, self).stats_fn(train_batch)

        additional = {}

        if self.config["enable_copo"]:
            key_list = ["lcf", "mean_nei_vf_loss", "mean_global_vf_loss", "normalized_advantages"]
            if self.config[USE_DISTRIBUTIONAL_LCF]:
                additional["lcf_std"] = torch.mean(torch.stack(self.get_tower_stats("lcf_std"))).detach()
            for k in ["step_lcf", GLOBAL_ADVANTAGES, NEI_ADVANTAGE, SampleBatch.ACTION_LOGP]:
                additional[k] = train_batch[k].mean()
        else:
            key_list = ["normalized_advantages"]

        for k in key_list:
            additional[k] = torch.mean(torch.stack(self.get_tower_stats(k)))

        ret.update(convert_to_numpy(additional))
        return ret

    def update_old_policy(self):
        model_state_dict = self.model.state_dict()
        self.target_model.load_state_dict(model_state_dict)

    def assign_lcf(self, lcf_parameters, lcf_mean, lcf_std=None, my_name=None):
        """
        Input: lcf_param in [-1, 1]
        Though we have updated LCF for the local policy. However, the LCF
        should be updated for all policies (including remote policies) since
        the postprocessing of trajectory is conducted in each policy separately.
        """
        lcf_parameters = lcf_parameters.to(self.device)
        old_mean = self.model.lcf_mean.item()
        assert self.model.lcf_parameters.size() == lcf_parameters.size()
        with torch.no_grad():
            self.model.lcf_parameters.data.copy_(lcf_parameters)
        assert torch.sum(self.model.lcf_parameters - lcf_parameters).item() < 1e-5, \
            (self.model.lcf_parameters, lcf_parameters, torch.sum(self.model.lcf_parameters - lcf_parameters).item())
        new_mean = self.model.lcf_mean.item()
        assert abs(new_mean - lcf_mean) < 1e-5, (new_mean, lcf_mean, new_mean - lcf_mean)
        new_std = None
        if lcf_std is not None:
            new_std = self.model.lcf_std.item()
            assert abs(new_std - lcf_std) < 1e-5, (new_std, lcf_std, new_std - lcf_std)
        if my_name is not None:
            print("In policy {}, latest LCF mean {}, std {}. Old mean {}. The update target {}".format(
                my_name, new_mean, new_std, old_mean, lcf_parameters
            ))

    def replaced_reward_for(self, batch):
        infos = batch.get(SampleBatch.INFOS)
        if self.config["randomized_dynamics"] and "dynamics_parameters" not in batch:
            batch["dynamics_parameters"] = np.stack(
                [dynamics_parameters_to_embedding(info["dynamics"]) for info in infos]).astype(np.float32)

        if hasattr(batch, "_reward_replaced") and batch._reward_replaced:
            return batch

        # latent_tensor = convert_to_torch_tensor(batch["latent"], device=self.device) \
        #     if self.config["enable_latent"] else None


        if isinstance(self.action_space, Box):
            clipped_actions = np.clip(batch["actions"], self.action_space.low[0], self.action_space.high[0])
        else:
            clipped_actions = action_discrete_to_continuous(
                batch["actions"], discrete_action_dim=self.config["discrete_action_dim"]
            )

        reward = self.discriminator.get_reward(
            obs=convert_to_torch_tensor(batch[SampleBatch.OBS], device=self.device),
            act=convert_to_torch_tensor(clipped_actions, device=self.device),
            next_obs=convert_to_torch_tensor(batch[SampleBatch.NEXT_OBS], device=self.device),
            action_logp=convert_to_torch_tensor(batch[SampleBatch.ACTION_LOGP], device=self.device),
            latent=None,
            discriminator_reward_native=self.config["discriminator_reward_native"],
            # add_action=self.config["discriminator_add_action"],
            gamma=self.config["gamma"]
        )
        reward = convert_to_numpy(reward).astype(batch[SampleBatch.REWARDS].dtype)
        native_reward = batch[SampleBatch.REWARDS]
        batch[SampleBatch.REWARDS] = np.clip(reward, -100, 100) + native_reward * self.config[
            "reward_augmentation_weight"]
        batch._reward_replaced = True
        return batch

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        infos = sample_batch.get(SampleBatch.INFOS)
        # if self.config["enable_latent"]:
        #     sample_batch["latent"] = sample_batch[SampleBatch.OBS][:, :self.config["latent_dim"]]

        if episode is not None and self.config["randomized_dynamics"]:
            sample_batch["dynamics_parameters"] = np.stack(
                [dynamics_parameters_to_embedding(info["dynamics"]) for info in infos]).astype(np.float32)

        if episode is not None:
            sample_batch["carsize"] = np.stack([info["carsize"] for info in infos]).astype(np.float32)

        with torch.no_grad():
            # ========== MODIFICATION!!! ==========
            if self.config["enable_discriminator"] and episode is not None:
                sample_batch = self.replaced_reward_for(sample_batch)
                assert sample_batch._reward_replaced
            # ========== MODIFICATION!!! ==========

            # Phase 1: Process the possible centralized critics: (copied from CCPPO but replace rewards!)

            o = sample_batch[SampleBatch.CUR_OBS]
            odim = o.shape[1]

            if episode is None:
                # In initialization, we set centralized_critic_obs_dim
                self.centralized_critic_obs_dim = sample_batch[CENTRALIZED_CRITIC_OBS].shape[1]
            else:
                # After initialization, fill centralized obs
                sample_batch[CENTRALIZED_CRITIC_OBS] = np.zeros(
                    (o.shape[0], self.centralized_critic_obs_dim),
                    dtype=sample_batch[SampleBatch.CUR_OBS].dtype
                )
                sample_batch[CENTRALIZED_CRITIC_OBS][:, :odim] = o

                assert other_agent_batches is not None
                other_info_dim = odim
                adim = sample_batch[SampleBatch.ACTIONS].shape[1] if isinstance(self.action_space, Box) else None
                if self.config[COUNTERFACTUAL]:
                    other_info_dim += adim

                if self.config["fuse_mode"] == "concat":
                    sample_batch = concat_ccppo_process(
                        self, sample_batch, other_agent_batches, odim, adim, other_info_dim
                    )
                elif self.config["fuse_mode"] == "mf":
                    sample_batch = mean_field_ccppo_process(
                        self, sample_batch, other_agent_batches, odim, adim, other_info_dim
                    )
                elif self.config["fuse_mode"] == "none":
                    # Do nothing since OBS is already filled
                    assert odim == sample_batch[CENTRALIZED_CRITIC_OBS].shape[1]
                else:
                    raise ValueError("Unknown fuse mode: {}".format(self.config["fuse_mode"]))

            # Use centralized critic to compute the value
            sample_batch[SampleBatch.VF_PREDS] = self.model.central_value_function(
                convert_to_torch_tensor(sample_batch[CENTRALIZED_CRITIC_OBS], self.device)
            ).cpu().detach().numpy().astype(np.float32)

            if sample_batch[SampleBatch.DONES][-1]:
                last_r = 0.0
            else:
                last_r = sample_batch[SampleBatch.VF_PREDS][-1]
            sample_batch = compute_advantages(
                sample_batch,
                last_r,
                self.config["gamma"],
                self.config["lambda"],
                use_gae=self.config["use_gae"],
                use_critic=self.config.get("use_critic", True)
            )

            # Phase 2: Fill CoPO advantages / values
            cobs = convert_to_torch_tensor(sample_batch[CENTRALIZED_CRITIC_OBS], self.device)

            if self.config["enable_copo"]:
                sample_batch[NEI_VALUES] = self.model.get_nei_value(cobs).cpu().detach().numpy().astype(np.float32)
                sample_batch[GLOBAL_VALUES] = self.model.get_global_value(cobs).cpu().detach().numpy().astype(
                    np.float32)
                if episode is not None:  # After initialization
                    assert isinstance(infos[0], dict)

                    # ========== MODIFICATION!!! ==========
                    if self.config["enable_discriminator"]:
                        sample_batch[NEI_REWARDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
                        sample_batch[GLOBAL_REWARDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
                        for index in range(sample_batch.count):
                            environmental_time_step = sample_batch["t"][index]

                            # Compute neighborhood reward
                            nrews = []
                            for nei_count, nei_name in enumerate(sample_batch['infos'][index]["neighbours"]):
                                if nei_name in other_agent_batches:
                                    _, nei_batch = other_agent_batches[nei_name]

                                    match_its_step = np.where(nei_batch["t"] == environmental_time_step)[0]
                                    if len(match_its_step) == 0:
                                        # Can't find
                                        pass
                                    elif len(match_its_step) > 1:
                                        raise ValueError()
                                    else:
                                        new_index = match_its_step[0]
                                        if not hasattr(nei_batch, "_reward_replaced"):
                                            nei_batch = self.replaced_reward_for(nei_batch)
                                            assert nei_batch._reward_replaced
                                        nei_rew = nei_batch[SampleBatch.REWARDS][new_index]
                                        nrews.append(nei_rew)
                            if nrews:
                                sample_batch[NEI_REWARDS][index] = np.mean(nrews).astype(np.float32)

                            # Compute global reward
                            grews = []
                            for a in sample_batch['infos'][index]["all_agents"]:

                                # Other agents might have different starting step!
                                # So we need to find exact sample of other agent by
                                # matching the env time step.

                                if a in other_agent_batches:
                                    _, a_batch = other_agent_batches[a]

                                    match_its_step = np.where(a_batch["t"] == environmental_time_step)[0]
                                    if len(match_its_step) == 0:
                                        # Can't find
                                        pass
                                    elif len(match_its_step) > 1:
                                        raise ValueError()
                                    else:
                                        new_index = match_its_step[0]
                                        if not hasattr(a_batch, "_reward_replaced"):
                                            a_batch = self.replaced_reward_for(a_batch)
                                            assert a_batch._reward_replaced
                                        a_rew = a_batch[SampleBatch.REWARDS][new_index]
                                        grews.append(a_rew)
                            if grews:
                                sample_batch[GLOBAL_REWARDS][index] = np.mean(grews).astype(np.float32)

                    else:
                        sample_batch[NEI_REWARDS] = np.array([info[NEI_REWARDS] for info in infos]).astype(np.float32)
                        sample_batch[GLOBAL_REWARDS] = np.array([info[GLOBAL_REWARDS] for info in infos]).astype(
                            np.float32)
                    # ========== MODIFICATION!!! ==========

                    # Note: step_lcf is in [-1, 1]
                    sample_batch["step_lcf"] = np.array([info["lcf"] for info in infos]).astype(np.float32)

            if self.config["enable_copo"]:
                # ===== Compute native, neighbour and global advantage =====
                # Note: native advantage is computed in super()
                if sample_batch[SampleBatch.DONES][-1]:
                    last_global_r = last_nei_r = 0.0
                else:
                    last_global_r = sample_batch[GLOBAL_VALUES][-1]
                    last_nei_r = sample_batch[NEI_VALUES][-1]
                sample_batch = compute_nei_advantage(sample_batch, last_nei_r, self.config["gamma"],
                                                     self.config["lambda"])
                sample_batch = compute_global_advantage(
                    sample_batch, last_global_r, gamma=1.0, lambda_=self.config["lambda"]
                )
                # ========== Modification: This part should be run after training discriminator! ==========

                assert sample_batch["step_lcf"].max() == sample_batch["step_lcf"].min()

        return sample_batch

    def get_weights(self):
        native = {k: v.cpu().detach().numpy() for k, v in self.model.state_dict().items()}
        dis = {k: v.cpu().detach().numpy() for k, v in self.discriminator.state_dict().items()}
        return {
            "native": native,
            "dis": dis
        }

    def set_weights(self, weights) -> None:
        weights = convert_to_torch_tensor(weights, device=self.device)
        self.model.load_state_dict(weights["native"])
        self.discriminator.load_state_dict(weights["dis"])


# ========== Load the states of real trajectories ==========


def _get_latent(latent_dim):
    # return np.random.normal(0, 1, latent_dim)
    raise ValueError()


def load_human_data(enable_latent, latent_dim=None, enable_latent_posterior=False, data_manager=None, debug=False):
    import pickle

    human_data_dir = osp.join(NEWCOPO_DIR, "2023-02-06_generate_waymo_data/waymo_human_states_0206")

    obs_list = []  # dataset size, obs dim
    next_obs_list = []  # dataset size, obs dim
    carsize_list = []  # dataset size, 2
    unique_list = []
    trajectory_dict = {}

    file_list = sorted(os.listdir(human_data_dir), key=lambda s: int(s.replace(".pkl", "")))
    if debug:
        file_list = file_list[:10]
    print("===== Loading Data ===== Begin! Loading from {} files.".format(len(file_list)))

    for p in tqdm(file_list, desc="Loading the state sequences from real data"):
        if not p.endswith(".pkl"):
            continue
        case_id = int(p.replace(".pkl", ""))
        current_traffic_data = data_manager.get_case(case_id, should_copy=False)
        carsize_dict = {}
        p = osp.join(human_data_dir, p)
        with open(p, "rb") as f:
            data = pickle.load(f)

        tmp_traj_dict = {}  # Key: vid

        for time_step in range(len(data)):
            if (time_step + 1 not in data) or (not data[time_step]):
                continue
            for v_id, v_state in data[time_step].items():
                if v_id not in carsize_dict:
                    # Tracks = dict(vid: dict(state: np.ndarray, type: str))
                    # state = [200, 10]
                    # Take the first timestep of the state, and the index 4 is the car width
                    carsize_id = v_id
                    if carsize_id == "sdc":
                        carsize_id = current_traffic_data["sdc_index"]
                    width = current_traffic_data['tracks'][carsize_id]['state'][0][4]
                    length = current_traffic_data['tracks'][carsize_id]['state'][0][3]
                    carsize_dict[v_id] = [width / 10, length / 10]  # Match the scale factor in env.

                if v_id not in tmp_traj_dict:
                    tmp_traj_dict[v_id] = []

                tmp_traj_dict[v_id].append(v_state)

                if v_id in data[time_step + 1]:
                    o = v_state
                    next_o = data[time_step + 1][v_id]
                    if enable_latent:
                        latent = np.zeros([latent_dim, ])
                        o = np.concatenate([latent, o], axis=0)
                        next_o = np.concatenate([latent, next_o], axis=0)
                    unique = case_id * 100000 + (int(v_id) if v_id != "sdc" else 0)
                    unique_list.append(unique)
                    carsize_list.append(carsize_dict[v_id])
                    obs_list.append(o)
                    next_obs_list.append(next_o)

            assert len(next_obs_list) == len(obs_list)

        for v_id, v_traj in tmp_traj_dict.items():
            unique = case_id * 100000 + (int(v_id) if v_id != "sdc" else 0)
            trajectory_dict[unique] = np.stack(v_traj)


        # print("===== Loading Data ===== Finished case {}. Current data size {}.".format(case_id, len(obs_list)))

    # Merge all data into a sample batch
    human_data = {
        SampleBatch.OBS: np.stack(obs_list),
        SampleBatch.NEXT_OBS: np.stack(next_obs_list),
        "carsize": np.stack(carsize_list, axis=0),
        "unique": np.array(unique_list),
    }

    human_data = SampleBatch(human_data)

    print("===== Loading Data ===== Finished! Current data size {}.".format(human_data.count))
    return human_data, trajectory_dict


def load_copo_data(enable_latent, latent_dim=None, discard_useless=True, enable_latent_posterior=False, data_manager=None, traj_len=None):
    from ray.rllib.offline import JsonReader
    # jr = JsonReader(os.path.join(NEWCOPO_DIR, "2023-01-13_best_copo_checkpoint_dataset"))
    # jr = JsonReader(os.path.join(NEWCOPO_DIR, "2023-02-23_train_copo/2023-02-23_collect_copo_dataset_randomized_dy"))

    # jr = JsonReader(os.path.join(NEWCOPO_DIR, "2023-02-23_train_copo/2023-02-23_best_copo_dataset_backup"))
    jr = JsonReader(os.path.join(NEWCOPO_DIR, "2023-02-23_train_copo/2023-02-23_best_copo_dataset"))

    train_batch = SampleBatch.concat_samples(list(jr.read_all_files()))

    # For DEBUG
    # train_batch = SampleBatch.concat_samples([
    #     next(jr.read_all_files()),
    #     next(jr.read_all_files())
    # ])

    print("===== Loading CoPO Data ===== Finished! Current data size {}.".format(train_batch.count))

    # This is for the case when you are using "naive" randomized dynamics. We will have 3 modes.
    mode_list = []
    dynamics_list = []
    # latent_dict = {}
    for index in range(train_batch.count):
        mode_list.append(train_batch['infos'][index]['dynamics_mode'])
        dy = train_batch['infos'][index]['dynamics']
        dy = dynamics_parameters_to_embedding(dy)
        dynamics_list.append(dy)

    train_batch["mode"] = np.array(mode_list)
    train_batch["dynamics"] = np.array(dynamics_list)

    if enable_latent:
        unique_list = []
        carsize_list = []

        # Read all possible data to collect a list of carsize.
        seed_set = set()
        for index in range(train_batch.count):
            seed = train_batch['infos'][index]['environment_seed']
            seed_set.add(seed)

        carsize_dict = {}
        # sdc_index_dict = {}
        for seed in seed_set:
            current_traffic_data = data_manager.get_case(seed, should_copy=False)
            carsize_dict[seed] = {}
            for carsize_id, v_data in current_traffic_data["tracks"].items():
                width = v_data['state'][0][4]
                length = v_data['state'][0][3]
                carsize_dict[seed][carsize_id] = [width / 10, length / 10]  # Match the scale factor in env.
            carsize_dict[seed]["sdc"] = carsize_dict[seed][current_traffic_data["sdc_index"]]
            # sdc_index_dict[seed] = current_traffic_data["sdc_index"]

        for index in range(train_batch.count):
            seed = train_batch['infos'][index]['environment_seed']
            # if seed not in latent_dict:
            #     latent_dict[seed] = {}
            vid = train_batch['infos'][index]['vehicle_id']
            unique_list.append(
                seed * 100000 + (int(vid) if vid != "sdc" else 0)
            )
            carsize_list.append(carsize_dict[seed][vid])

        # train_batch["latent"] = np.stack(latent_list)
        train_batch["carsize"] = np.stack(carsize_list)
        train_batch["unique"] = np.stack(unique_list)

    trajectory_dict = {}

    episode_dict = split_batch_to_episodes(train_batch)
    for unique, episode in episode_dict.items():
        # Every "episode" is a trajectory of one agent
        traj = episode["obs"]

        assert traj.shape[1] == 280
        """
        [0, 10) - latent
        [10, 130) - side detector
        [130, 148) - state info
        [148, 158) - navi info
        [158, 159) - lateral dist
        [159, 279) - lidar
        [279, 280) - lcf

        total - 280
        """

        # traj = traj[:, 130:159]

        # Return full trajectory
        traj = traj[:, 10:279]

        trajectory_dict[unique] = traj

    deleted_keys = set(train_batch.keys())

    # Keep these keys and discard all others, to save memory
    deleted_keys.discard("obs")
    deleted_keys.discard("new_obs")
    deleted_keys.discard("actions")
    deleted_keys.discard("carsize")
    deleted_keys.discard("unique")
    deleted_keys.discard("mode")

    if enable_latent:
        deleted_keys.discard("latent")

    if discard_useless:
        for k in deleted_keys:
            del train_batch[k]

    return train_batch, trajectory_dict


# ========== The New Trainer ==========
class MultiAgentIRL(CCPPOTrainer):
    @classmethod
    def get_default_config(cls):
        return NewConfig()

    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        assert config["framework"] == "torch"
        return NewPolicy

    def generate_mixed_batch_for_discriminator(self, train_batch, policy):

        data_size = train_batch.policy_batches["default"]["obs"].shape[0]
        enable_latent = self.config["enable_latent"]
        enable_latent_posterior = self.config["enable_latent_posterior"]
        latent_dim = self.config["latent_dim"]

        if self.human_data_count + data_size >= self.human_data.count:
            self.human_data.shuffle()
            self.human_data_count = 0

        data_type = np.float32

        expert_batch = self.human_data.slice(self.human_data_count, self.human_data_count + data_size)

        print("Generating expert data with index [{}, {}]".format(
            self.human_data_count, self.human_data_count + data_size))

        expert_batch = expert_batch.copy()
        expert_batch["labels"] = np.ones([expert_batch.count, ], dtype=data_type)

        agent_batch = {
            SampleBatch.OBS: train_batch.policy_batches["default"][SampleBatch.OBS].astype(data_type),
            SampleBatch.NEXT_OBS: train_batch.policy_batches["default"][SampleBatch.NEXT_OBS].astype(data_type),
            "labels": np.zeros([train_batch.policy_batches["default"][SampleBatch.OBS].shape[0], ], dtype=data_type),
            "carsize": train_batch.policy_batches["default"]["carsize"].astype(data_type),
        }

        if enable_latent_posterior:
            # Fill expert dataset with latest estimated latent

            max_trajectory_len = self.config["trajectory_len"]
            uniques = expert_batch['unique']

            latent_added_obs = []
            latent_added_new_obs = []

            for ind, u in enumerate(uniques):
                expert_traj = self.expert_traj_dict[u]

                expert_state, expert_timestep, expert_mask = _pad_a_traj(expert_traj, max_trajectory_len)
                latent = policy.discriminator.get_latent(
                    convert_to_torch_tensor(expert_timestep.unsqueeze(0), device=policy.device),
                    convert_to_torch_tensor(expert_state.unsqueeze(0), device=policy.device),
                    convert_to_torch_tensor(expert_mask.unsqueeze(0), device=policy.device),
                )[0]
                latent = convert_to_numpy(latent)

                if self.config["use_copo_dataset"]:
                    latent_added_obs.append(np.concatenate([
                        latent, expert_batch["obs"][ind, 10:]
                    ], axis=-1))

                    latent_added_new_obs.append(np.concatenate([
                        latent, expert_batch["new_obs"][ind, 10:]
                    ], axis=-1))

                else:
                    latent_added_obs.append(np.concatenate([
                        latent, expert_batch["obs"][ind, latent_dim:]
                    ], axis=-1))

                    latent_added_new_obs.append(np.concatenate([
                        latent, expert_batch["new_obs"][ind, latent_dim:]
                    ], axis=-1))

            expert_batch["obs"] = np.stack(latent_added_obs)
            expert_batch["new_obs"] = np.stack(latent_added_new_obs)


        agent_batch = SampleBatch(agent_batch)

        if self.config["algorithm"] in ["airl"]:
            agent_batch[SampleBatch.ACTIONS] = \
                train_batch.policy_batches["default"][SampleBatch.ACTIONS].astype(data_type)

        if self.config["use_copo_dataset"] and self.config["enable_copo"]:
            # Discard LCF in agent batch
            agent_batch[SampleBatch.OBS] = agent_batch[SampleBatch.OBS][:, :-1]
            agent_batch[SampleBatch.NEXT_OBS] = agent_batch[SampleBatch.NEXT_OBS][:, :-1]
            # expert_batch[SampleBatch.OBS] = expert_batch[SampleBatch.OBS][:, :-1]
            # expert_batch[SampleBatch.NEXT_OBS] = expert_batch[SampleBatch.NEXT_OBS][:, :-1]

        if self.config["use_copo_dataset"]:
            expert_batch[SampleBatch.OBS] = expert_batch[SampleBatch.OBS][:, :-1]
            expert_batch[SampleBatch.NEXT_OBS] = expert_batch[SampleBatch.NEXT_OBS][:, :-1]
            if not enable_latent:
                expert_batch[SampleBatch.OBS] = expert_batch[SampleBatch.OBS][:, latent_dim:]
                expert_batch[SampleBatch.NEXT_OBS] = expert_batch[SampleBatch.NEXT_OBS][:, latent_dim:]

            expert_batch.pop(SampleBatch.ACTIONS)

        if self.config["algorithm"] in ["airl"]:
            # Fill the actions into the dataset
            with torch.no_grad():
                pseudo_action = policy.discriminator.compute_pseudo_action(
                    obs=convert_to_torch_tensor(expert_batch[SampleBatch.OBS], device=policy.device),
                    next_obs=convert_to_torch_tensor(expert_batch[SampleBatch.NEXT_OBS], device=policy.device),
                    low=policy.action_space.low[0] if isinstance(policy.action_space, Box) else None,
                    high=policy.action_space.high[0] if isinstance(policy.action_space, Box) else None,
                    latent=None,
                    # latent=convert_to_torch_tensor(expert_batch["latent"], device=policy.device) \
                    #     if self.config["enable_latent"] else None,
                    # latent=None,

                    # TODO: When computing actions for expert using inverse, we might not know dynamics.
                    #  It is possible we call the dynamics model to get an estimated dynamics.
                    #  Or we can simply use latent code. This might need more tests.
                    # dynamics_parameters=convert_to_torch_tensor(train_batch["dynamics_parameters"], device=self.device),
                    dynamics_parameters=None
                )

                if isinstance(policy.action_space, Box):
                    # raise ValueError("AIRL only supports discrete action space.")
                    expert_batch[SampleBatch.ACTIONS] = convert_to_numpy(pseudo_action).astype(data_type)
                else:
                    expert_batch[SampleBatch.ACTIONS] = convert_to_numpy(pseudo_action.max(1)[1]).astype(data_type)

            # Note: Different to GAIL, in AIRL we should have LCF in obs because we need to query policy!
            # We should fill LCF into the obs of expert states!
            if self.config["enable_copo"]:
                lcf_list = np.clip(np.random.normal(
                    policy.model.lcf_mean.item(),  # in [-1, 1]
                    policy.model.lcf_std.item(),
                    (expert_batch.count, 1)
                ), -1, 1)
                lcf_list: np.ndarray = (lcf_list + 1) / 2  # scale to [0, 1]
                lcf_list = lcf_list.astype(data_type)

                expert_batch.intercepted_values.clear()
                expert_batch[SampleBatch.OBS] = np.concatenate(
                    [expert_batch[SampleBatch.OBS], lcf_list], axis=1
                ).astype(data_type)
                expert_batch[SampleBatch.NEXT_OBS] = np.concatenate(
                    [expert_batch[SampleBatch.NEXT_OBS], lcf_list], axis=1
                ).astype(data_type)

                native_obs_len = 270  # Hardcoded!!!!!!!!!!!!

            else:
                native_obs_len = 269

            if enable_latent:
                assert agent_batch[SampleBatch.OBS].shape[1] == native_obs_len + latent_dim
                assert expert_batch[SampleBatch.OBS].shape[1] == native_obs_len + latent_dim
            else:
                assert agent_batch[SampleBatch.OBS].shape[1] == native_obs_len == expert_batch["obs"].shape[1]

        elif self.config["algorithm"] in ["gail"]:
            if not self.config["use_copo_dataset"] and self.config["enable_copo"]:
                # Discard LCF in agent batch
                agent_batch[SampleBatch.OBS] = agent_batch[SampleBatch.OBS][:, :-1]
                agent_batch[SampleBatch.NEXT_OBS] = agent_batch[SampleBatch.NEXT_OBS][:, :-1]

            native_obs_len = 269  # Hardcoded here! Tired!
            if enable_latent:
                assert agent_batch[SampleBatch.OBS].shape[1] == native_obs_len + latent_dim
                assert expert_batch[SampleBatch.OBS].shape[1] == native_obs_len + latent_dim
            else:
                assert agent_batch[SampleBatch.OBS].shape[1] == native_obs_len == expert_batch["obs"].shape[1]

        else:
            raise ValueError()

        try:
            expert_batch = SampleBatch({k: convert_to_numpy(expert_batch[k]).astype(data_type) for k in agent_batch.keys()})
            agent_batch.intercepted_values.clear()
            assert isinstance(expert_batch["obs"], np.ndarray)
            assert isinstance(agent_batch["obs"], np.ndarray)

            if enable_latent:
                assert np.all(expert_batch["obs"][:, :latent_dim] == expert_batch["new_obs"][:, :latent_dim])
                if not np.all(agent_batch["obs"][:, :latent_dim] == agent_batch["new_obs"][:, :latent_dim]):
                    # In rare case, the `obs` might be sampled in the end of last training iteration.
                    # At that time, the posterior model is not updated yet and thus the latent stored in the
                    # `obs` are the result of old posterior model.
                    agent_batch["obs"][:, :latent_dim] = agent_batch["new_obs"][:, :latent_dim]
                # assert np.all(agent_batch["new_obs"][:, :latent_dim] == agent_batch["latent"])

            dis_batch = SampleBatch.concat_samples([expert_batch, agent_batch])

            if self.config["use_copo_dataset"] and self.config["algorithm"] in ["airl"]:
                dis_batch[SampleBatch.ACTIONS] = dis_batch[SampleBatch.ACTIONS].astype(int)

        except Exception as e:
            print(expert_batch["obs"].shape, agent_batch["obs"].shape, expert_batch["obs"], agent_batch["obs"])
            raise e

        if self.config["algorithm"] in ["airl"]:
            with torch.no_grad():
                dis_batch = policy._lazy_tensor_dict(dis_batch, device=policy.device)
                logits, state = policy.model(dis_batch)
                curr_action_dist = policy.dist_class(logits, policy.model)

            dis_batch[SampleBatch.ACTION_LOGP] = curr_action_dist.logp(dis_batch[SampleBatch.ACTIONS])

        return dis_batch

    def _set_dynamics_parameters_distribution_to_environment(self):
        local_policy = self.get_policy("default")

        # Sync Dynamics Parameters with Envs for next training iteration
        if self.config["randomized_dynamics"] in ["mean", "std"]:
            d_mean, d_log_std = local_policy.discriminator.get_dynamics_parameters_distribution(
                device=local_policy.device)
            d_mean = d_mean.cpu().detach().numpy()
            d_std = np.exp(np.clip(d_log_std.cpu().detach().numpy(), -20, 10))

            def _set_dynamics(e):
                e.set_dynamics_parameters_distribution(dynamics_parameters_mean=d_mean,
                                                       dynamics_parameters_std=d_std)

            self.workers.foreach_env(_set_dynamics)

        elif self.config["randomized_dynamics"] in ["gmm"]:
            enable_latent = self.config["enable_latent"]
            enable_latent_posterior = self.config["enable_latent_posterior"]
            latent_dim = self.config["latent_dim"]
            latent_dict = self.latent_dict

            def _set_dynamics_function_for_worker(wid, w):
                local_policy = w.get_policy("default")

                def dynamics_function(*, environment_seed, agent_name, **kwargs):
                    if enable_latent:
                        unique = environment_seed * 100000 + (int(agent_name) if agent_name != "sdc" else 0)
                        if unique not in latent_dict:
                            latent = np.random.normal(0, 1, (latent_dim,))
                        else:
                            latent = latent_dict[unique]
                        latent = convert_to_torch_tensor(latent, device=local_policy.device).reshape(1, -1)
                    else:
                        latent = None
                    with torch.no_grad():
                        dynamics_matrix, weights = local_policy.discriminator.get_dynamics_parameters_distribution(
                            nn_batch_size=1,
                            latent=latent,
                            device=local_policy.device
                        )
                        one_hot = gumbel_softmax(logits=weights, temperature=1.0, device=local_policy.device)
                        selected_dynamics = one_hot @ dynamics_matrix
                        # in [bs, dynamics dim * 2]

                        loc, loc_std = torch.chunk(selected_dynamics, 2, dim=-1)
                        d_std = torch.exp(loc_std.clamp(-20, 10))
                        dist = Normal(loc, d_std)
                        dynamics_parameters = dist.sample()

                    ret = convert_to_numpy(dynamics_parameters)[0]
                    return ret, {}

                w.foreach_env(lambda e: e.set_dynamics_parameters_distribution(dynamics_function=dynamics_function))

            # Fill the batch to this worker
            self.workers.foreach_worker_with_id(_set_dynamics_function_for_worker)

        elif self.config["randomized_dynamics"] in ["nn", "nnstd"]:

            if self.config["enable_latent"]:
                latent_dim = self.config["latent_dim"]

                def _set_dynamics_function_for_worker(wid, w):
                    local_policy = w.get_policy("default")

                    def dynamics_function(*, environment_seed, agent_name, latent_dict, **kwargs):
                        if environment_seed not in latent_dict:
                            latent_dict[environment_seed] = {}
                        if agent_name not in latent_dict[environment_seed]:
                            # For some reason the agent is filterer out in the human dataset
                            # But it shows up in the environment.
                            # In this case, we create a latent for it on the fly.
                            # It would be good if we sync this with other environments.
                            # But for simplicity, we just ignore this part.
                            # Therefore, it is possible that two different MD env creates two latents for one trajectory
                            # in the dataset.
                            latent_dict[environment_seed][agent_name] = _get_latent(latent_dim)
                            # print("Create a latent for Worker {}, Env {}, Agent {}".format(
                            #     wid, environment_seed, agent_name))

                        latent = latent_dict[environment_seed][agent_name]
                        ret = local_policy.discriminator.get_dynamics_parameters_distribution(
                            nn_batch_size=1,
                            latent=convert_to_torch_tensor(latent, device=local_policy.device).reshape(1, -1),
                            device=local_policy.device
                        )
                        ret = convert_to_numpy(ret)[0]
                        return ret, {}

                    w.foreach_env(lambda e: e.set_dynamics_parameters_distribution(dynamics_function=dynamics_function))

                # Fill the batch to this worker
                self.workers.foreach_worker_with_id(_set_dynamics_function_for_worker)

            else:
                # Set local worker
                d_mean = local_policy.discriminator.get_dynamics_parameters_distribution(
                    nn_batch_size=200,
                    device=local_policy.device
                )
                d_mean = convert_to_numpy(d_mean)
                # in [-1, 1]

                # Fill the batch to this worker
                self.workers.local_worker().foreach_env(
                    lambda e: e.set_dynamics_parameters_distribution(dynamics_parameters_mean=d_mean))

                # print("WORKER {}, DISCRIMINATOR WEIGHT {}".format(
                #     0,
                #     list(self.workers.local_worker().get_policy(
                #         "default").discriminator._dynamics_model.state_dict().values())[0][0]
                # ))
                #
                # Set remote workers
                for worker_id in range(self.workers.num_remote_workers()):
                    # Generate a batch
                    d_mean = local_policy.discriminator.get_dynamics_parameters_distribution(
                        nn_batch_size=200,
                        device=local_policy.device
                    )
                    d_mean = convert_to_numpy(d_mean)

                    # In [-1, 1]

                    # print("Worker {}, Param {}".format(worker_id + 1, d_mean[0]))

                    # Create a function
                    def _set_dynamics(wid, w):
                        # print("WORKER {}, DISCRIMINATOR WEIGHT {}".format(
                        #     wid,
                        #     list(w.get_policy(
                        #         "default").discriminator._dynamics_model.state_dict().values())[0][0]
                        # ))

                        w.foreach_env(
                            lambda e: e.set_dynamics_parameters_distribution(dynamics_parameters_mean=d_mean)
                        )
                        # print("Finished setting worker {} with {}".format(wid, d_mean[0]))

                    # Fill the batch to this worker
                    self.workers.foreach_worker_with_id(_set_dynamics, local_worker=False,
                                                        remote_worker_ids=[worker_id + 1])


        elif self.config["randomized_dynamics"] in [DYNAMICS_POLICY]:

            latent_dict = self.latent_dict
            latent_dim = self.config["latent_dim"]

            def _set_dynamics_function_for_worker(wid, w):
                local_policy = w.get_policy("default")

                def dynamics_function(*, environment_seed, agent_name, **kwargs):
                    unique = environment_seed * 100000 + (int(agent_name) if agent_name != "sdc" else 0)
                    if unique not in latent_dict:
                        latent = np.random.normal(0, 1, (latent_dim,))
                    else:
                        latent = latent_dict[unique]
                    ret = local_policy.discriminator.get_dynamics_parameters_distribution(
                        nn_batch_size=1,
                        latent=convert_to_torch_tensor(latent, device=local_policy.device).reshape(1, -1),
                        device=local_policy.device
                    )
                    ret = convert_to_numpy(ret)[0]
                    return ret, {}

                w.foreach_env(
                    lambda e: e.set_dynamics_parameters_distribution(dynamics_function=dynamics_function))

            # Fill the batch to this worker
            self.workers.foreach_worker_with_id(_set_dynamics_function_for_worker)

        else:
            assert self.config["randomized_dynamics"] is None

    # def __getstate__(self):
    #     ret = super().__getstate__()
    #     if self.config["enable_latent"]:
    #         ret["worker"]["latent_dict"] = self.latent_dict
    #     return ret

    def _set_latent_function_to_environment(self):
        assert self.config["enable_latent"]
        assert self.config["enable_latent_posterior"]

        # latent_trajectory_len = self.config["latent_trajectory_len"]

        latent_dict = self.latent_dict
        latent_dim = self.config["latent_dim"]

        def _set_latent_function_to_worker(wid, w):
            # local_policy = w.get_policy("default")


            # print("WORKER {}, DISCRIMINATOR POSTERIOR MODEL WEIGHT {}".format(
            #     wid,
            #     list(local_policy.discriminator._posterior_model.state_dict().values())[0][0]
            # ))

            def _latent_function(seed, vid):
                unique = seed * 100000 + (int(vid) if vid != "sdc" else 0)
                if unique not in latent_dict:
                    latent = np.random.normal(0, 1, (latent_dim,))
                else:
                    latent = latent_dict[unique]
                return latent

            w.foreach_env(lambda e: e.register_latent_function(latent_function=_latent_function))

        # Fill the batch to this worker
        self.workers.foreach_worker_with_id(_set_latent_function_to_worker)

    # def generate_expert_batch_for_posterior(self, data_size, minibatch_size, device, num_samples):
    def generate_expert_batch_for_posterior(self, expert_traj_dict, agent_traj_dict, device, num_neg_samples=1):
        max_trajectory_len = self.config["trajectory_len"]


        batch = dict(
            expert_state=[], expert_timestep=[], expert_mask=[],
            agent_state=[], agent_timestep=[], agent_mask=[],
        )
        for unique_id, agent_traj in agent_traj_dict.items():
            if unique_id not in expert_traj_dict:
                continue
            expert_traj = expert_traj_dict[unique_id]

            expert_state, expert_timestep, expert_mask = _pad_a_traj(expert_traj, max_trajectory_len)
            batch["expert_state"].append(expert_state)
            batch["expert_timestep"].append(expert_timestep)
            batch["expert_mask"].append(expert_mask)

            agent_state, agent_timestep, agent_mask = _pad_a_traj(agent_traj, max_trajectory_len)
            batch["agent_state"].append(agent_state)
            batch["agent_timestep"].append(agent_timestep)
            batch["agent_mask"].append(agent_mask)

        batch = {k: convert_to_torch_tensor(torch.stack(v), device=device) for k, v in batch.items()}
        batch = SampleBatch(batch)

        # Build another sample batch for training on the expert trajectory
        expert_batch = dict(
            origin_state=[], origin_timestep=[], origin_mask=[],
            pos_state=[], pos_timestep=[], pos_mask=[],
            # contain_pos=[],
            # contain_neg=[],
        )
        for x in range(num_neg_samples):
            expert_batch["neg_state_{}".format(x)] =[]
            expert_batch["neg_timestep_{}".format(x)] =[]
            expert_batch["neg_mask_{}".format(x)] =[]


        id_list = list(expert_traj_dict.keys())
        for unique_id, expert_traj in expert_traj_dict.items():

            if expert_traj.shape[0] >= max_trajectory_len:  # Worth to create a new positive sample
                expert_state, expert_timestep, expert_mask = _pad_a_traj(expert_traj, max_trajectory_len)
                expert_batch["pos_state"].append(expert_state)
                expert_batch["pos_timestep"].append(expert_timestep)
                expert_batch["pos_mask"].append(expert_mask)
                # expert_batch["contain_pos"].append(True)
            else:
                continue

            for x in range(num_neg_samples):
                neg_id = unique_id
                while neg_id == unique_id:
                    neg_id = random.choice(id_list)
                neg_state, neg_timestep, neg_mask = _pad_a_traj(expert_traj_dict[neg_id], max_trajectory_len)
                expert_batch["neg_state_{}".format(x)].append(neg_state)
                expert_batch["neg_timestep_{}".format(x)].append(neg_timestep)
                expert_batch["neg_mask_{}".format(x)].append(neg_mask)
                # expert_batch["contain_neg"].append(True)

            expert_state, expert_timestep, expert_mask = _pad_a_traj(expert_traj, max_trajectory_len)
            expert_batch["origin_state"].append(expert_state)
            expert_batch["origin_timestep"].append(expert_timestep)
            expert_batch["origin_mask"].append(expert_mask)
        new_expert_batch = {}
        for k, v in expert_batch.items():
            if "contain" in k:
                new_expert_batch[k] = convert_to_torch_tensor(torch.from_numpy(np.array(v)), device=device)
            else:
                new_expert_batch[k] = convert_to_torch_tensor(torch.stack(v), device=device)
        expert_batch = SampleBatch(new_expert_batch)
        return batch, expert_batch


    def generate_agent_trajectory_dict(self, train_batch, should_slice):
        trajectory_dict = {}
        batch_dict = {}
        episode_dict = split_batch_to_episodes(train_batch.policy_batches["default"])
        for unique, episode in episode_dict.items():
            # Every "episode" is a trajectory of one agent
            traj = episode["obs"]
            """
            [0, 10) - latent
            [10, 130) - side detector
            [130, 148) - state info
            [148, 158) - navi info
            [158, 159) - lateral dist
            [159, 279) - lidar
            [279, 280) - lcf

            total - 280
            """
            if should_slice:
                if traj.shape[1] == 279:
                    traj = traj[:, 130:159]
                elif traj.shape[1] == 280:
                    traj = traj[:, 130:159]
                elif traj.shape[1] == 269:
                    traj = traj[:, 120:149]
                else:
                    raise ValueError()
            else:
                traj = traj[:, self.config["latent_dim"]:]
            trajectory_dict[unique] = traj
            batch_dict[unique] = episode
        return trajectory_dict, batch_dict

    def generate_forward_batch(self, policy, agent_batch_dict, max_trajectory_len):
        latent_dim = self.config["latent_dim"]
        gamma = self.config["gamma"]

        # Fill reward here:
        batch = dict(
            states=[], timesteps=[], traj_mask=[], actions=[], dynamics=[], next_states=[]
        )
        for unique_id, agent_episode in agent_batch_dict.items():
            if unique_id not in self.latent_dict:
                continue
            states, timesteps, traj_mask, actions, dynamics, next_state, action_logp = _pad_a_traj_for_forward_model(
                agent_episode, max_trajectory_len, latent_dim=self.config["latent_dim"], device=policy.device
            )
            batch["states"].append(states)
            batch["next_states"].append(next_state)
            batch["timesteps"].append(timesteps)
            batch["traj_mask"].append(traj_mask)
            batch["actions"].append(actions)
            batch["dynamics"].append(dynamics)

        batch = {k: convert_to_torch_tensor(torch.stack(v), device=policy.device) for k, v in batch.items()}
        batch = SampleBatch(batch)
        return batch


    def generate_dynamics_reward(self, policy, agent_batch_dict, train_batch):
        latent_dim = self.config["latent_dim"]
        gamma = self.config["gamma"]

        # Fill reward here:
        batch = dict(
            latent=[], dynamics_parameters=[], reward=[], action_logp=[]
        )
        for unique_id, agent_episode in agent_batch_dict.items():

            if unique_id not in self.latent_dict:
                continue

            latent = self.latent_dict[unique_id]

            logit = policy.discriminator._dynamics_model(convert_to_torch_tensor(latent.reshape(1, -1), device=policy.device))
            loc, log_std = torch.chunk(logit, 2, dim=1)

            std = torch.exp(log_std.clamp(-20, 10))
            dist = Normal(loc, std)
            dynamics_parameters = dist.sample()
            logp = dist.log_prob(dynamics_parameters).sum(-1)

            dynamics_parameters = dynamics_parameters.clamp(-1, 1)

            # state = convert_to_torch_tensor(agent_episode["obs"], device=policy.device)
            # real_next_state = convert_to_torch_tensor(agent_episode["new_obs"], device=policy.device)

            # if isinstance(policy.action_space, Box):
            #     clipped_actions = np.clip(agent_episode["actions"], policy.action_space.low[0], policy.action_space.high[0])
            # else:
            #     clipped_actions = action_discrete_to_continuous(
            #         agent_episode["actions"], discrete_action_dim=self.config["discrete_action_dim"]
            #     )
            #
            # action = convert_to_torch_tensor(clipped_actions, device=policy.device)
            # action_logp = convert_to_torch_tensor(agent_episode[SampleBatch.ACTION_LOGP], device=policy.device)

            # fake_next_obs = policy.discriminator.get_next_state(
            #     state,
            #     action,
            #     dynamics_parameters.tile([agent_episode.count, 1]), latent=None
            # )

            state, timesteps, mask, action, dynamics, next_state, action_logp = _pad_a_traj_for_forward_model(
                agent_episode, self.config["trajectory_len"], latent_dim=self.config["latent_dim"], device=policy.device
            )

            fake_next_obs, action, dynamics = policy.discriminator.get_next_state(
                timesteps.reshape(-1, *timesteps.shape),
                state.reshape(-1, *state.shape),
                action.reshape(-1, *action.shape),
                dynamics.reshape(-1, *dynamics.shape),
                mask.reshape(-1, *mask.shape),
            )

            fake_next_obs = fake_next_obs.clamp(0, 1)  # [1, max_traj_len, 269]

            fake_next_obs = fake_next_obs[0]  # Reshape to [max_traj_len, 269]

            if self.config["enable_latent"]:
                latent_batch = convert_to_torch_tensor(latent, device=policy.device).tile([fake_next_obs.shape[0], 1])
                fake_next_obs = torch.cat([latent_batch, fake_next_obs], dim=1)
                next_state = torch.cat([latent_batch, next_state], dim=1)
                state = torch.cat([latent_batch, state], dim=1)

            if state.shape[1] == 270:
                state = state[:, :-1]

            reward = policy.discriminator.get_reward(obs=state, next_obs=fake_next_obs, action_logp=action_logp)
            real_reward = policy.discriminator.get_reward(obs=state, next_obs=next_state, action_logp=action_logp)

            reward = reward - real_reward  # Use the reward differences as the improvement brought by the new dynamics

            reward = reward.mean()  # Average the reward across the whole trajectory

            batch["latent"].append(latent)
            batch["reward"].append(reward)
            batch["dynamics_parameters"].append(dynamics_parameters)
            batch["action_logp"].append(logp)

        batch = SampleBatch(dict(
            latent=convert_to_torch_tensor(np.stack(batch["latent"]), device=policy.device),
            dynamics_parameters=torch.cat(batch["dynamics_parameters"]),
            reward=torch.stack(batch["reward"]),
            action_logp=torch.stack(batch["action_logp"])
        ))

        # Normalize
        batch["reward"] = standardized(batch["reward"])

        return batch


    @ExperimentalAPI
    def training_step(self):
        local_policy = self.get_policy("default")

        # Load data first!
        if not hasattr(self, "human_data"):
            if self.config["use_copo_dataset"]:
                self.human_data, expert_traj_dict = load_copo_data(
                    enable_latent=self.config["enable_latent"],
                    latent_dim=self.config["latent_dim"],
                    enable_latent_posterior=self.config["enable_latent_posterior"],
                    data_manager=self.workers.local_worker().env.engine.data_manager,
                )
                self.expert_traj_dict = expert_traj_dict
            else:
                while self.workers.local_worker().env is None:
                    import time
                    time.sleep(0.5)
                    print("Sleep...")
                self.human_data, expert_traj_dict = load_human_data(
                    enable_latent=self.config["enable_latent"],
                    latent_dim=self.config["latent_dim"],
                    enable_latent_posterior=self.config["enable_latent_posterior"],
                    data_manager=self.workers.local_worker().env.engine.data_manager,
                    debug=self.config["debug"]
                )
                print("Successfully load human data.")
                self.expert_traj_dict = expert_traj_dict

            self.human_data.shuffle()
            self.human_data_count = 0

            # if self.config["enable_latent"] and not self.config["enable_latent_posterior"]:
            #     self.workers.foreach_env(lambda e: e.register_latent(latent_dict))
            #     self.latent_dict = latent_dict


        # ===== Before sample data in the environment, sync the dynamics to all envs =====
        if self.config["enable_latent"] and self.config["enable_latent_posterior"]:
            self.latent_dict = local_policy.discriminator.compute_latent_dict(
                self.expert_traj_dict, self.config["trajectory_len"], device=local_policy.device
            )
            # Update the latent function at each training step!
            self._set_latent_function_to_environment()
        self._set_dynamics_parameters_distribution_to_environment()

        # ===== Collect SampleBatches from sample workers until we have a full batch. =====
        if self.config.count_steps_by == "agent_steps":
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.config.train_batch_size
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config.train_batch_size
            )

        # ===== Build agent's trajectories!!! =====
        # We do this here because we don't want the training data to be shuffled!
        agent_trajectory_dict, agent_batch_dict = self.generate_agent_trajectory_dict(train_batch, should_slice=self.config["randomized_dynamics"] != DYNAMICS_POLICY)

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        assert len(train_batch.policy_batches) == 1
        recorder = defaultdict(list)

        if self.config["algorithm"] in ["airl"]:
            # Train the inverse dynamic model first!
            for i in range(self.config["inverse_num_iters"]):
                for mb_id, minibatch in enumerate(
                        minibatches(train_batch.policy_batches["default"],
                                    self.config.inverse_sgd_minibatch_size)):
                    ret = local_policy.update_inverse_model(minibatch)
                    for k, v in ret.items():
                        recorder[k].append(v)

                # print("===== Inverse Model Training {} Iterations ===== Stats: {}".format(
                #     i, {k: round(sum(v) / len(v), 4) for k, v in recorder.items() if "inverse" in k}
                # ))

        # Fill expert data into the batch
        dis_batch = self.generate_mixed_batch_for_discriminator(train_batch, local_policy)
        if self.config['algorithm'] in ["airl"]:
            recorder["mixed_batch_logp_mean"] = dis_batch[SampleBatch.ACTION_LOGP].mean()
            recorder["mixed_batch_logp_min"] = dis_batch[SampleBatch.ACTION_LOGP].min()

        # ===== Update discriminator =====
        for i in range(self.config["discriminator_num_iters"]):
            for mb_id, minibatch in enumerate(
                    minibatches(dis_batch, self.config["discriminator_sgd_minibatch_size"])):
                ret = local_policy.update_discriminator(minibatch)
                for k, v in ret.items():
                    recorder[k].append(v)
            # print("===== Discriminator Training {} Iterations ===== Stats: {}".format(
            #     i, {k: round(sum(v) / len(v), 4) for k, v in recorder.items() if "inverse" not in k}
            # ))

        if self.config["randomized_dynamics"] and self.config["randomized_dynamics"] != DYNAMICS_POLICY:
            # Update Forward Model (Use inverse model's hyper)
            for i in range(self.config["inverse_num_iters"]):
                for mb_id, minibatch in enumerate(
                        minibatches(train_batch.policy_batches["default"], self.config.inverse_sgd_minibatch_size)):
                    ret = local_policy.update_forward_model(minibatch)
                    for k, v in ret.items():
                        recorder[k].append(v)

                # print("===== Forward Model Training {} Iterations ===== Stats: {}".format(
                #     i, {k: round(sum(v) / len(v), 4) for k, v in recorder.items() if "forward" in k}
                # ))

            # Update Dynamics Parameters Distribution
            first_iter_dynamics_scores = 0.0
            for i in range(self.config["dynamics_num_iters"]):
                for mb_id, minibatch in enumerate(
                        minibatches(train_batch.policy_batches["default"], self.config.discriminator_sgd_minibatch_size)
                ):

                    # TODO: We should update dynamics to "confuse discriminator"
                    ret = local_policy.update_dynamics(minibatch)
                    for k, v in ret.items():
                        recorder[k].append(v)

                    if i == 0:
                        first_iter_dynamics_scores = ret["dynamics_scores"]
                    if i == self.config["dynamics_num_iters"] - 1:
                        last_iter_dynamics_scores = ret["dynamics_scores"]
                        recorder["dynamics_scores_before"] = first_iter_dynamics_scores
                        recorder["dynamics_scores_after"] = last_iter_dynamics_scores
                        recorder["dynamics_scores_diff"] = last_iter_dynamics_scores - first_iter_dynamics_scores
                    # print("===== Dynamics Training {} Iterations ===== Stats: {}".format(
                    #     i, {k: round(sum(v) / len(v), 4) for k, v in recorder.items() if "dynamics" in k}
                    # ))

        # ===== Update the dynamics policy here =====
        if self.config["randomized_dynamics"] == DYNAMICS_POLICY:
            # Update Forward Model (Use inverse model's hyper)

            # Create a new batch for training forward model.
            forward_batch = self.generate_forward_batch(
                policy=local_policy, agent_batch_dict=agent_batch_dict,
                max_trajectory_len=self.config["trajectory_len"],
            )
            for i in range(self.config["inverse_num_iters"]):
                for mb_id, minibatch in enumerate(minibatches(forward_batch, self.config.inverse_sgd_minibatch_size)):
                    ret = local_policy.update_forward_model(minibatch)
                    for k, v in ret.items():
                        recorder[k].append(v)

            first_iter_dynamics_scores = 0.0
            for i in range(self.config["dynamics_num_iters"]):

                dynamics_batch = self.generate_dynamics_reward(train_batch=train_batch, policy=local_policy, agent_batch_dict=agent_batch_dict)

                for mb_id, minibatch in enumerate(
                        minibatches(dynamics_batch, self.config.discriminator_sgd_minibatch_size)
                ):

                    # TODO: We should update dynamics to "confuse discriminator"
                    ret = local_policy.update_dynamics_policy(minibatch)
                    for k, v in ret.items():
                        recorder[k].append(v)

                    if i == 0:
                        first_iter_dynamics_scores = ret["dynamics_scores"]
                    if i == self.config["dynamics_num_iters"] - 1:
                        last_iter_dynamics_scores = ret["dynamics_scores"]
                        recorder["dynamics_scores_before"] = first_iter_dynamics_scores
                        recorder["dynamics_scores_after"] = last_iter_dynamics_scores
                        recorder["dynamics_scores_diff"] = last_iter_dynamics_scores - first_iter_dynamics_scores


        # ===== Update Latent Code =====
        data_size = train_batch.policy_batches["default"]["obs"].shape[0]
        if self.config["enable_latent_posterior"]:
            posterior_batch, contrastive_batch = self.generate_expert_batch_for_posterior(
                # data_size, self.config["discriminator_sgd_minibatch_size"], local_policy.device,
                # self.config["posterior_num_samples"]
                self.expert_traj_dict, agent_trajectory_dict, local_policy.device,
                num_neg_samples=self.config["num_neg_samples"]
            )
            for i in range(self.config["discriminator_num_iters"]):
                # for mb_id, minibatch in enumerate(
                #         minibatches(posterior_batch, self.config["discriminator_sgd_minibatch_size"])
                # ):
                ret = local_policy.update_posterior(
                    posterior_batch,
                    contrastive_batch,
                    self.config["posterior_match_loss_weight"],
                    self.config["num_neg_samples"],
                )
                for k, v in ret.items():
                    recorder[k].append(v)

                # It's time to update the posterior_target!
                local_policy.discriminator.update_posterior_target()

        self.human_data_count += data_size

        recorder = convert_to_numpy(recorder)
        discriminator_stats = {k: np.mean(v) for k, v in recorder.items()}

        # ===== Training the policy =====
        # ===== Update some CoPO modules =====
        if self.config["enable_copo"]:
            for policy_id, batch in train_batch.policy_batches.items():
                used_lcf = batch["step_lcf"] * np.pi / 2
                batch["normalized_advantages"] = (
                        np.cos(used_lcf) * batch[Postprocessing.ADVANTAGES] + np.sin(used_lcf) * batch[NEI_ADVANTAGE]
                )
                batch["raw_normalized_advantages"] = batch["normalized_advantages"]

                # New: Just put the mean and std of advantage in local policy, will be used in Meta Update.
                local_policy._raw_lcf_adv_mean = batch["normalized_advantages"].mean()
                local_policy._raw_lcf_adv_std = max(1e-4, batch["normalized_advantages"].std())

                batch["normalized_advantages"] = standardized(batch["normalized_advantages"])
                batch[GLOBAL_ADVANTAGES] = standardized(batch[GLOBAL_ADVANTAGES])
        else:
            for policy_id, batch in train_batch.policy_batches.items():
                batch["normalized_advantages"] = standardized(batch[Postprocessing.ADVANTAGES])

        if self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        # Add stats for new reward
        rew = train_batch.policy_batches["default"]["rewards"]
        discriminator_stats["discriminator_reward_outcome_mean"] = rew.mean().item()
        discriminator_stats["discriminator_reward_outcome_max"] = rew.max().item()
        discriminator_stats["discriminator_reward_outcome_min"] = rew.min().item()

        policies_to_update = list(train_results.keys())

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.num_remote_workers() > 0:
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(
                    policies=policies_to_update,
                    global_vars=global_vars,
                )

        # ========== CoPO Modification ==========
        # Update LCF
        assert len(train_batch.policy_batches) == 1
        recorder = defaultdict(list)

        if self.config["enable_copo"]:
            lcf_sgd_minibatch_size = self.config["lcf_sgd_minibatch_size"] or self.config["sgd_minibatch_size"]
            for i in range(self.config["lcf_num_iters"]):
                for mb_id, minibatch in enumerate(
                        minibatches(train_batch.policy_batches["default"], lcf_sgd_minibatch_size)):
                    ret = local_policy.meta_update(minibatch)
                    for k, v in ret.items():
                        recorder[k].append(v)
            require_return_last_keys = ["lcf", "lcf_std"]
            recorder = convert_to_numpy(recorder)
            average_ret = {k: np.mean(v) for k, v in recorder.items() if k not in require_return_last_keys}
            ret = convert_to_numpy(ret)
            last_ret = {k: v for k, v in ret.items() if k in require_return_last_keys}

            lcf_parameters = local_policy.model.lcf_parameters.cpu().detach()
            lcf_mean = local_policy.model.lcf_mean.item()
            lcf_std = local_policy.model.lcf_std.item()

            def _update_lcf_2(w_id, w):
                def _update_lcf_1(pi, pi_id):
                    name = "{}_{}".format(w_id, pi_id)
                    pi.assign_lcf(lcf_parameters, lcf_mean, lcf_std, name)
                    pi.update_old_policy()

                w.foreach_policy(_update_lcf_1)

                def _set_env_lcf(e):
                    e.set_lcf_dist(mean=lcf_mean, std=lcf_std)

                w.foreach_env(_set_env_lcf)

            self.workers.foreach_worker_with_id(_update_lcf_2)
            meta_update_fetches = {}
            meta_update_fetches["raw_lcf_adv_mean_value"] = local_policy._raw_lcf_adv_mean
            meta_update_fetches["raw_lcf_adv_std_value"] = local_policy._raw_lcf_adv_std
            meta_update_fetches.update(average_ret)
            meta_update_fetches.update(last_ret)

            # Note: Since we share the same policy and other networks for all vehicles,
            # we have a shared "meta_update" (stat dict) here.
            # We put it into the custom metrics dict for the policy named "default_policy".
            train_results["default"]["meta_update"] = meta_update_fetches
            print("Current LCF in degree: ", meta_update_fetches["lcf"] * 90)

        train_results["default"].update(discriminator_stats)

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                    self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                    log_once("ppo_warned_lr_ratio")
                    and self.config.get("model", {}).get("vf_share_layers")
                    and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                    log_once("ppo_warned_vf_clip")
                    and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results


# ========== Test scripts ==========
def _test():
    # Testing only!
    from scenarionet_training.marl.utils.callbacks import MultiAgentDrivingCallbacks
    from scenarionet_training.marl.utils.train import train
    from scenarionet_training.marl.utils.utils import get_train_parser
    from ray import tune
    parser = get_train_parser()
    args = parser.parse_args()
    stop = {"timesteps_total": 3000}
    exp_name = "debug" if not args.exp_name else args.exp_name
    num_gpus = 1 if torch.cuda.is_available() else 0
    if num_gpus == 1:
        print("========== GPU is being used! ==========")
    config = dict(
        # env=get_rllib_compatible_env(get_latent_env(get_lcf_env(MARLWaymoEnv))),
        env=get_rllib_compatible_env(get_latent_posterior_env(get_lcf_env(MARLWaymoEnv))),
        env_config=dict(
            # Reward augmentation
            driving_reward=0,
            success_reward=0,
            out_of_road_penalty=1,
            crash_vehicle_penalty=1,
            crash_object_penalty=1,

            # discrete_action=tune.grid_search([True, False]),
            discrete_action=True,
            # discrete_action_dim=5,
            # start_seed=tune.grid_search([5000]),
            # num_agents=8,

            case_num=10,

            # randomized_dynamics=True,
        ),
        # num_sgd_iter=1,
        # rollout_fragment_length=200,
        # train_batch_size=400,
        # sgd_minibatch_size=200,

        # **{USE_CENTRALIZED_CRITIC: True},
        # fuse_mode=tune.grid_search(["concat", "mf"])
        # fuse_mode=tune.grid_search(["none"])

        num_workers=0,
        # algorithm=tune.grid_search(["airl", "gail"]),
        algorithm=tune.grid_search(["gail"]),

        train_batch_size=100,
        rollout_fragment_length="auto",
        sgd_minibatch_size=30,

        discriminator_l2=1e-5,
        discriminator_lr=1e-4,  # They use LR linear decay! We should change this too!
        discriminator_num_iters=3,
        discriminator_sgd_minibatch_size=30,

        enable_discriminator=True,
        discriminator_add_action=True,
        discriminator_reward_native=True,

        counterfactual=False,
        enable_copo=False,

        use_copo_dataset=False,
        # use_copo_dataset_with_inverse_model=False,

        reward_augmentation_weight=1,

        # randomized_dynamics=tune.grid_search([None, "mean", "std", "nn", "nnstd"]),
        # randomized_dynamics=tune.grid_search([None, "nn", "nnstd"]),
        # randomized_dynamics=tune.grid_search(["nnstd"]),
        # randomized_dynamics=tune.grid_search(["gmm", ]),
        randomized_dynamics=tune.grid_search(["dynamics_policy", ]),
        # dynamics_use_std=True

        # enable_latent=False,
        enable_latent=True,
        enable_latent_posterior=True,
        num_neg_samples=3,

        latent_dim=10,

        dynamics_posterior_weight=1.0,
        posterior_lr=1.0,
        # enable_latent=False

        num_gpus=num_gpus,
        create_env_on_local_worker=True,  # Have to create this!!!!!
        trajectory_len=20,

        debug=True
    )
    results = train(
        MultiAgentIRL,
        config=config,  # Do not use validate_config_add_multiagent here!
        checkpoint_freq=1,  # Don't save checkpoint is set to 0.
        keep_checkpoints_num=0,
        stop=stop,
        num_gpus=1 if num_gpus > 0 else 0,
        num_seeds=1,
        max_failures=0,
        exp_name=exp_name,
        custom_callback=MultiAgentDrivingCallbacks,
        test_mode=True,
        local_mode=True,

        # wandb_project="newcopo",
        # wandb_team="drivingforce",
    )


if __name__ == "__main__":

    s = time.time()
    _test()
    e = time.time()

    print("Overall time: ", e - s)

    # load_human_data()
