"""
Do not modify this file. This file is deprecated. Use maairl.py!
"""

import logging
import os
import os.path as osp
from collections import defaultdict
from typing import Type

import gym
import numpy as np
from gym.spaces import Box
from ray.rllib.evaluation.postprocessing import Postprocessing
# import torch
# import torch.nn as nn
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
from tqdm.auto import tqdm

from scenarionet_training.marl.algo.ccppo import CCPPOTrainer, CCPPOPolicy, CCPPOConfig
from scenarionet_training.marl.algo.ccppo import mean_field_ccppo_process, concat_ccppo_process
from scenarionet_training.marl.algo.copo import CoPOModel

ModelCatalog.register_custom_model("copo_model", CoPOModel)

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

torch, nn = try_import_torch()

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

# path to ~/newcopo/newcopo
NEWCOPO_DIR = osp.dirname(osp.dirname(osp.abspath(osp.dirname(__file__))))


class NewConfig(CCPPOConfig):
    # TODO: This is identical to the CoPO config! Change if needed
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or CCPPOTrainer)

        self.initial_lcf_std = 0.1
        self.lcf_sgd_minibatch_size = None
        self.lcf_num_iters = 5
        self.lcf_lr = 1e-4
        self.use_distributional_lcf = True
        self.use_centralized_critic = False
        self.fuse_mode = "none"
        self.old_value_loss = True
        self.update_from_dict({
            "model": {
                "custom_model": "copo_model",
            }})

        # ===== New Discriminator Config =====
        self.discriminator_l2 = 1e-5
        self.discriminator_lr = 5e-4  # 1e-4 is best for Waymo data
        self.discriminator_num_iters = 100  # 5 is best for Waymo data
        self.discriminator_sgd_minibatch_size = 1024
        self.inverse_num_iters = 100
        self.inverse_sgd_minibatch_size = 512
        self.inverse_lr = 1e-4

        # Important!
        self.enable_discriminator = True
        self.discriminator_reward_native = False
        self.discriminator_add_action = False
        self.discriminator_use_tanh = False

        self.use_copo_dataset = False
        self.use_copo_dataset_with_inverse_model = False  # Set these two = True to use inverse model for CoPO dataset

    def validate(self):
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

class DiscriminatorDiscrete(TorchModelV2, nn.Module):
    def __init__(
            self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
            model_config: ModelConfigDict, name: str
    ):

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        assert not isinstance(action_space, Box)

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

        # ========== Our Modification: We compute the centralized critic obs size here! ==========
        # === Reward Model ===
        prev_layer_size = 270  # Hardcoded!!!
        if model_config["discriminator_add_action"]:
            raise ValueError()
            # prev_layer_size += int(np.product(action_space.shape))

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
        prev_layer_size = 270  # Hardcoded!!!
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
        prev_layer_size = 270 * 2  # Hardcoded!!!
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
            out_size=action_space.n,
            initializer=normc_initializer(0.01),
            activation_fn="tanh" if model_config["discriminator_use_tanh"] else None
        ))
        self._inverse_model = nn.Sequential(*layers)
        self._inverse_loss = torch.nn.CrossEntropyLoss()

        # === Forward Model (but without action) ===
        forward_model_latent_dim = 64
        # Part 1: Encoder
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        for size in [256, ]:
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
            out_size=forward_model_latent_dim * 2,  # VAE!
            initializer=normc_initializer(0.01),
            activation_fn=None
        ))
        self._forward_model_encoder = nn.Sequential(*layers)
        # Part 2: Decoder
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        for size in [forward_model_latent_dim, 256]:
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
            out_size=int(np.prod(obs_space.shape)),
            initializer=normc_initializer(0.01),
            activation_fn=None
        ))
        self._forward_model_decoder = nn.Sequential(*layers)
        self._forward_model_recon_loss = torch.nn.MSELoss()

    def get_discriminator_parameters(self):
        return list(self._reward_model.parameters()) + list(self._value_model.parameters())

    def get_inverse_parameters(self):
        return list(self._inverse_model.parameters())

    def get_internal_reward(self, *, obs, act, add_action):

        # if obs.shape[1] == 270:
        #     obs = obs[:, :-1]

        if add_action:
            rew_input = torch.cat([obs, act], dim=-1)
        else:
            rew_input = obs
        r = self._reward_model(rew_input)
        r = r.reshape(-1)
        return r

    def get_reward(self, *, obs, act=None, next_obs=None, action_logp=None, discriminator_reward_native=False,
                   add_action=True, gamma=None):

        rew = self.get_internal_reward(obs=obs, act=act, add_action=add_action)

        if not discriminator_reward_native:
            # Just return the reward if not using native AIRL
            return rew

        assert next_obs is not None
        assert gamma is not None
        assert action_logp is not None

        next_value = self.get_value(next_obs)
        value = self.get_value(obs)

        log_p_tau = rew + gamma * next_value - value

        log_q_tau = action_logp

        log_p_tau = log_p_tau.clamp(-20, 10).reshape(-1)
        log_q_tau = log_q_tau.clamp(-20, 10).reshape(-1)

        sum_and_exp = torch.exp(log_p_tau) + torch.exp(log_q_tau)

        log_pq = torch.log(sum_and_exp.clamp(1e-6, 10e6))

        scores = torch.exp(log_p_tau - log_pq)

        # scores should be in [0, 1]
        scores = torch.clamp(scores, 1e-6, 1 - 1e-6)
        ret = torch.log(scores) - torch.log(1 - scores)

        return ret

    def get_value(self, obs):

        # if obs.shape[1] == 270:
        #     obs = obs[:, :-1]

        ret = self._value_model(obs)
        ret = ret.reshape(-1)
        # print("Return value: ", ret.shape)
        return ret

    def compute_inverse_loss(self, obs, next_obs, actions, low, high):
        pseudo_action = self.compute_pseudo_action(obs, next_obs, low=low, high=high)
        inv_loss = self._inverse_loss(input=pseudo_action, target=actions.long())
        return inv_loss, pseudo_action

    def compute_pseudo_action(self, obs, next_obs, low, high):
        # inv_input = torch.cat([obs[:, :150], next_obs[:, :150]], dim=-1)

        # if obs.shape[1] == 270:
        #     obs = obs[:, :-1]

        # if next_obs.shape[1] == 270:
        #     next_obs = next_obs[:, :-1]

        inv_input = torch.cat([obs, next_obs], dim=-1)
        pseudo_action = self._inverse_model(inv_input)

        if self.model_config["discriminator_use_tanh"]:
            raise ValueError()
            # pseudo_action = (pseudo_action + 1) / 2 * (high - low) + low
            # assert pseudo_action.max().item() <= high
            # assert pseudo_action.min().item() >= low

        return pseudo_action

    def compute_discriminator_loss(self, *, obs, act, next_obs, action_logp, labels, gamma, add_action):

        log_q_tau = action_logp

        reward = self.get_internal_reward(obs=obs, act=act, add_action=add_action)

        next_value = self.get_value(next_obs)
        value = self.get_value(obs)

        log_p_tau = reward + gamma * next_value - value

        log_p_tau = log_p_tau.clamp(-20, 10).reshape(-1)
        log_q_tau = log_q_tau.clamp(-20, 10).reshape(-1)

        sum_and_exp = torch.exp(log_p_tau) + torch.exp(log_q_tau)

        log_pq = torch.log(sum_and_exp.clamp(1e-6, 10e6))

        # log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)

        # self.discrim_output = tf.exp(log_p_tau - log_pq)

        loss = -torch.mean(
            labels * (log_p_tau - log_pq) + (1 - labels) * (log_q_tau - log_pq)
        )

        # Note: L2 loss is replaced by the weight decay in Adam optimizer.
        # self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params]) * self.l2_loss_ratio

        with torch.no_grad():
            pred = torch.exp(log_p_tau - log_pq)
            # Pred -> 1, should be agent sample, should labels=0.
            acc = torch.sum(((pred < 0.5) == labels)) / obs.shape[0]

        stat = {"reward": reward, "value": value, "accuracy": acc.item()}
        return loss, stat




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

        if isinstance(self.action_space, Box):
            self.discriminator = Discriminator(
                obs_space=observation_space,
                action_space=action_space,
                num_outputs=logit_dim,
                model_config=config["model"],
                name="discriminator"
            )

        else:
            assert not self.config[COUNTERFACTUAL]
            self.discriminator = DiscriminatorDiscrete(
                obs_space=observation_space,
                action_space=action_space,
                num_outputs=logit_dim,
                model_config=config["model"],
                name="discriminator"
            )

        self.discriminator.to(self.device)

        self._discriminator_optimizer = torch.optim.Adam(
            self.discriminator.get_discriminator_parameters(),
            lr=self.config["discriminator_lr"],
            weight_decay=self.config["discriminator_l2"]  # 0.1 in MA-AIRL impl.
        )

        self._inverse_optimizer = torch.optim.Adam(
            self.discriminator.get_inverse_parameters(),
            lr=self.config["inverse_lr"],  # Just reuse the hyperparameters!!!
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
            high=self.action_space.high[0] if isinstance(self.action_space, Box) else None
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

        discriminator_loss, stat = self.discriminator.compute_discriminator_loss(
            obs=convert_to_torch_tensor(train_batch[SampleBatch.OBS], device=self.device),
            act=convert_to_torch_tensor(train_batch[SampleBatch.ACTIONS], device=self.device),
            next_obs=convert_to_torch_tensor(train_batch[SampleBatch.NEXT_OBS], device=self.device),
            action_logp=convert_to_torch_tensor(train_batch[SampleBatch.ACTION_LOGP], device=self.device),
            labels=convert_to_torch_tensor(train_batch["labels"], device=self.device),
            gamma=self.config["gamma"],
            add_action=self.config["discriminator_add_action"]
        )
        self._discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self._discriminator_optimizer.step()

        reward = stat["reward"]
        value = stat["value"]

        stats = {
            "discriminator_loss": discriminator_loss.item(),
            "discriminator_reward_max": reward.max().item(),
            "discriminator_reward_mean": reward.mean().item(),
            "discriminator_reward_min": reward.min().item(),
            "discriminator_value_max": value.max().item(),
            "discriminator_value_mean": value.mean().item(),
            "discriminator_value_min": value.min().item(),
            "discriminator_accuracy": stat["accuracy"]
        }

        return stats

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

        # Modification: newly add stats
        model.tower_stats["lcf"] = model.lcf_mean
        if self.config[USE_DISTRIBUTIONAL_LCF]:
            model.tower_stats["lcf_std"] = model.lcf_std
        model.tower_stats["mean_nei_vf_loss"] = nei_mean_vf_loss
        model.tower_stats["mean_global_vf_loss"] = global_mean_vf_loss
        model.tower_stats["normalized_advantages"] = advantages.mean()
        return total_loss

    def stats_fn(self, train_batch):
        ret = super(NewPolicy, self).stats_fn(train_batch)

        additional = {}
        for k in [
            "lcf", "mean_nei_vf_loss", "mean_global_vf_loss", "normalized_advantages",
        ]:
            additional[k] = torch.mean(torch.stack(self.get_tower_stats(k)))

        if self.config[USE_DISTRIBUTIONAL_LCF]:
            additional["lcf_std"] = torch.mean(torch.stack(self.get_tower_stats("lcf_std"))).detach()

        for k in ["step_lcf", GLOBAL_ADVANTAGES, NEI_ADVANTAGE, SampleBatch.ACTION_LOGP]:
            additional[k] = train_batch[k].mean()

        ret.update(convert_to_numpy(additional))

        return ret


    #
    # def extra_grad_info(self, train_batch):
    #     ret = super(NewPolicy, self).extra_grad_info(train_batch)
    #
    #     additional_ret = {}
    #     for k in ["lcf", "mean_nei_vf_loss", "mean_global_vf_loss", "normalized_advantages"]:
    #         s = self.get_tower_stats(k)
    #         additional_ret[k] = torch.mean(torch.stack(s))
    #
    #     if self.config[USE_DISTRIBUTIONAL_LCF]:
    #         additional_ret["lcf_std"] = torch.mean(torch.stack(self.get_tower_stats("lcf_std")))
    #
    #     additional_ret = convert_to_numpy(additional_ret)
    #     ret.update(additional_ret)
    #     assert "entropy" in ret, ret.keys()
    #     return ret

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
        if hasattr(batch, "_reward_replaced") and batch._reward_replaced:
            return batch

        reward = self.discriminator.get_reward(
            obs=convert_to_torch_tensor(batch[SampleBatch.OBS], device=self.device),
            act=convert_to_torch_tensor(batch[SampleBatch.ACTIONS], device=self.device),
            next_obs=convert_to_torch_tensor(batch[SampleBatch.NEXT_OBS], device=self.device),
            action_logp=convert_to_torch_tensor(batch[SampleBatch.ACTION_LOGP], device=self.device),
            discriminator_reward_native=self.config["discriminator_reward_native"],
            add_action=self.config["discriminator_add_action"],
            gamma=self.config["gamma"]
        )
        reward = convert_to_numpy(reward).astype(batch[SampleBatch.REWARDS].dtype)
        batch[SampleBatch.REWARDS] = np.clip(reward, -100, 100)
        batch._reward_replaced = True
        return batch

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):

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

            # ========== Modification: This part should be run after training discriminator! ==========
            sample_batch[NEI_VALUES] = self.model.get_nei_value(cobs).cpu().detach().numpy().astype(np.float32)
            sample_batch[GLOBAL_VALUES] = self.model.get_global_value(cobs).cpu().detach().numpy().astype(np.float32)

            infos = sample_batch.get(SampleBatch.INFOS)
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
                    sample_batch[GLOBAL_REWARDS] = np.array([info[GLOBAL_REWARDS] for info in infos]).astype(np.float32)
                # ========== MODIFICATION!!! ==========

                # Note: step_lcf is in [-1, 1]
                sample_batch["step_lcf"] = np.array([info["lcf"] for info in infos]).astype(np.float32)

            # ===== Compute native, neighbour and global advantage =====
            # Note: native advantage is computed in super()
            if sample_batch[SampleBatch.DONES][-1]:
                last_global_r = last_nei_r = 0.0
            else:
                last_global_r = sample_batch[GLOBAL_VALUES][-1]
                last_nei_r = sample_batch[NEI_VALUES][-1]
            sample_batch = compute_nei_advantage(sample_batch, last_nei_r, self.config["gamma"], self.config["lambda"])
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
def load_human_data():
    import pickle

    human_data_dir = osp.join(NEWCOPO_DIR, "waymo_human_states_0202")

    obs_list = []  # dataset size, obs dim
    next_obs_list = []  # dataset size, obs dim

    file_list = sorted(os.listdir(human_data_dir))
    print("===== Loading Data ===== Begin! Loading from {} files.".format(len(file_list)))

    for p in tqdm(file_list, desc="Loading the state sequences from real data"):

        if not p.endswith(".pkl"):
            continue

        case_id = p.replace(".pkl", "")

        p = osp.join(human_data_dir, p)

        with open(p, "rb") as f:
            data = pickle.load(f)

        for time_step in range(len(data)):

            if (time_step + 1 not in data) or (not data[time_step]):
                continue

            for v_id, v_state in data[time_step].items():
                if v_id in data[time_step + 1]:
                    obs_list.append(v_state)
                    next_obs_list.append(data[time_step + 1][v_id])

            assert len(next_obs_list) == len(obs_list)

        # print("===== Loading Data ===== Finished case {}. Current data size {}.".format(case_id, len(obs_list)))

    # Merge all data into a sample batch

    ret = SampleBatch({
        SampleBatch.OBS: np.stack(obs_list),
        SampleBatch.NEXT_OBS: np.stack(next_obs_list)
    })
    print("===== Loading Data ===== Finished! Current data size {}.".format(ret.count))
    return ret


def load_copo_data():
    from ray.rllib.offline import JsonReader
    jr = JsonReader(os.path.join(NEWCOPO_DIR, "2023-01-13_best_copo_checkpoint_dataset"))
    train_batch = SampleBatch.concat_samples(list(jr.read_all_files()))
    print("===== Loading CoPO Data ===== Finished! Current data size {}.".format(train_batch.count))

    deleted_keys = set(train_batch.keys())
    deleted_keys.discard("obs")
    deleted_keys.discard("new_obs")
    deleted_keys.discard("actions")
    for k in deleted_keys:
        del train_batch[k]

    return train_batch


# ========== The New Trainer ==========
class NewTrainer(CCPPOTrainer):
    @classmethod
    def get_default_config(cls):
        return NewConfig()

    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        assert config["framework"] == "torch"
        return NewPolicy

    def generate_mixed_batch_for_discriminator(self, train_batch, policy):
        if not hasattr(self, "human_data"):

            if self.config["use_copo_dataset"]:
                self.human_data = load_copo_data()
            else:
                self.human_data = load_human_data()
            self.human_data.shuffle()
            self.human_data_count = 0

        data_size = train_batch.policy_batches["default"]["obs"].shape[0]
        if self.human_data_count + data_size >= self.human_data.count:
            self.human_data.shuffle()
            self.human_data_count = 0

        data_type = np.float32

        expert_batch = self.human_data.slice(self.human_data_count, self.human_data_count + data_size)

        print("Generating expert data with index [{}, {}]".format(
            self.human_data_count, self.human_data_count + data_size))

        self.human_data_count += data_size

        expert_batch = expert_batch.copy()
        expert_batch["labels"] = np.ones([expert_batch.count, ], dtype=data_type)

        if self.config["use_copo_dataset"]:
            if self.config["use_copo_dataset_with_inverse_model"]:

                # Fill the actions into the dataset
                with torch.no_grad():
                    expert_batch = policy._lazy_tensor_dict(expert_batch, device=policy.device)
                    pseudo_action = policy.discriminator.compute_pseudo_action(
                        obs=expert_batch[SampleBatch.OBS],
                        next_obs=expert_batch[SampleBatch.NEXT_OBS],
                        low=policy.action_space.low[0] if isinstance(policy.action_space, Box) else None,
                        high=policy.action_space.high[0] if isinstance(policy.action_space, Box) else None,
                    )

                    if isinstance(policy.action_space, Box):
                        raise ValueError("AIRL only supports discrete action space.")
                        expert_batch[SampleBatch.ACTIONS] = convert_to_numpy(pseudo_action).astype(data_type)
                    else:
                        expert_batch[SampleBatch.ACTIONS] = convert_to_numpy(pseudo_action.max(1)[1]).astype(data_type)

        else:
            # TODO: Interesting! We should fill LCF into the obs of expert states!
            lcf_list = np.clip(np.random.normal(
                policy.model.lcf_mean.item(),  # in [-1, 1]
                policy.model.lcf_std.item(),
                (expert_batch.count, 1)
            ), -1, 1)
            lcf_list: np.ndarray = (lcf_list + 1) / 2  # scale to [0, 1]
            lcf_list = lcf_list.astype(data_type)

            expert_batch.intercepted_values.clear()
            expert_batch[SampleBatch.OBS] = np.concatenate([expert_batch[SampleBatch.OBS], lcf_list], axis=1).astype(
                data_type)
            expert_batch[SampleBatch.NEXT_OBS] = np.concatenate([expert_batch[SampleBatch.NEXT_OBS], lcf_list],
                                                                axis=1).astype(data_type)

            # Fill the actions into the dataset
            with torch.no_grad():
                expert_batch = policy._lazy_tensor_dict(expert_batch, device=policy.device)
                pseudo_action = policy.discriminator.compute_pseudo_action(
                    obs=expert_batch[SampleBatch.OBS],
                    next_obs=expert_batch[SampleBatch.NEXT_OBS],
                    low=policy.action_space.low[0] if isinstance(policy.action_space, Box) else None,
                    high=policy.action_space.high[0] if isinstance(policy.action_space, Box) else None,
                )

                if isinstance(policy.action_space, Box):
                    expert_batch[SampleBatch.ACTIONS] = convert_to_numpy(pseudo_action).astype(data_type)
                else:
                    expert_batch[SampleBatch.ACTIONS] = convert_to_numpy(pseudo_action.max(1)[1]).astype(data_type)



        agent_batch = SampleBatch({
            SampleBatch.OBS: train_batch.policy_batches["default"][SampleBatch.OBS].astype(data_type),
            SampleBatch.NEXT_OBS: train_batch.policy_batches["default"][SampleBatch.NEXT_OBS].astype(data_type),
            SampleBatch.ACTIONS: train_batch.policy_batches["default"][SampleBatch.ACTIONS].astype(data_type),
        })
        agent_batch["labels"] = np.zeros([expert_batch.count, ], dtype=data_type)

        # if self.config["use_copo_dataset"]:
        #     # Discard LCF in agent batch
        #     agent_batch[SampleBatch.OBS] = agent_batch[SampleBatch.OBS][:, :-1]
        #     agent_batch[SampleBatch.NEXT_OBS] = agent_batch[SampleBatch.NEXT_OBS][:, :-1]
        #
        # else:
        #     # Discard LCF in agent batch
        #     agent_batch[SampleBatch.OBS] = agent_batch[SampleBatch.OBS][:, :-1]
        #     agent_batch[SampleBatch.NEXT_OBS] = agent_batch[SampleBatch.NEXT_OBS][:, :-1]

        native_obs_len = 270  # Hardcoded!!!!!!!!!!!!
        # Note: Different to GAIL, in AIRL we should have LCF in obs because we need to query policy!
        assert agent_batch[SampleBatch.OBS].shape[1] == native_obs_len == expert_batch["obs"].shape[1]

        try:
            expert_batch = SampleBatch({k: convert_to_numpy(v).astype(data_type) for k, v in expert_batch.items()})
            agent_batch.intercepted_values.clear()
            assert isinstance(expert_batch["obs"], np.ndarray)
            assert isinstance(agent_batch["obs"], np.ndarray)
            dis_batch = SampleBatch.concat_samples([expert_batch, agent_batch])

            if self.config["use_copo_dataset"]:
                dis_batch[SampleBatch.ACTIONS] = dis_batch[SampleBatch.ACTIONS].astype(int)

        except Exception as e:
            print(expert_batch["obs"].shape, agent_batch["obs"].shape, expert_batch["obs"], agent_batch["obs"])
            raise e

        with torch.no_grad():
            dis_batch = policy._lazy_tensor_dict(dis_batch, device=policy.device)
            logits, state = policy.model(dis_batch)
            curr_action_dist = policy.dist_class(logits, policy.model)

        dis_batch[SampleBatch.ACTION_LOGP] = curr_action_dist.logp(dis_batch[SampleBatch.ACTIONS])

        return dis_batch

    @ExperimentalAPI
    def training_step(self):
        # Collect SampleBatches from sample workers until we have a full batch.
        if self.config.count_steps_by == "agent_steps":
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.config.train_batch_size
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config.train_batch_size
            )
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        local_policy = self.get_policy("default")

        # ========== Discriminator Modification ==========

        # tensor_batch = self._lazy_tensor_dict(train_batch, device=self.device)

        # Train the inverse dynamic model first!
        assert len(train_batch.policy_batches) == 1
        recorder = defaultdict(list)
        for i in range(self.config["inverse_num_iters"]):
            for mb_id, minibatch in enumerate(
                    minibatches(train_batch.policy_batches["default"],
                                self.config.inverse_sgd_minibatch_size)):
                ret = local_policy.update_inverse_model(minibatch)
                for k, v in ret.items():
                    recorder[k].append(v)

            print("===== Inverse Model Training {} Iterations ===== Stats: {}".format(
                i, {k: round(sum(v) / len(v), 4) for k, v in recorder.items() if "inverse" in k}
            ))

        # Fill expert data into the batch
        dis_batch = self.generate_mixed_batch_for_discriminator(train_batch, local_policy)

        # Update discriminator
        for i in range(self.config["discriminator_num_iters"]):
            for mb_id, minibatch in enumerate(
                    minibatches(dis_batch, self.config["discriminator_sgd_minibatch_size"])):
                ret = local_policy.update_discriminator(minibatch)
                for k, v in ret.items():
                    recorder[k].append(v)
            print("===== Discriminator Training {} Iterations ===== Stats: {}".format(
                i, {k: round(sum(v) / len(v), 4) for k, v in recorder.items() if "inverse" not in k}
            ))

        # Modify the reward and advantage stored in the batch!
        # train_batch = local_policy.refill_training_batch(train_batch)

        recorder = convert_to_numpy(recorder)
        discriminator_stats = {k: np.mean(v) for k, v in recorder.items()}

        # ========== Discriminator Modification Ends ==========

        # ========== CoPO Modification ==========
        # Standardize advantages
        # train_batch = standardize_fields(train_batch, ["advantages"])  # Native PPO

        # wrapped = False
        # if isinstance(samples, SampleBatch):
        #     samples = MultiAgentBatch({"default": samples}, samples.count)
        #     wrapped = True
        assert len(train_batch.policy_batches) == 1
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
        # ========== CoPO Modification Ends ==========

        # Train
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
        lcf_sgd_minibatch_size = self.config["lcf_sgd_minibatch_size"] or self.config["sgd_minibatch_size"]
        recorder = defaultdict(list)
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

        train_results["default"].update(discriminator_stats)

        print("Current LCF in degree: ", meta_update_fetches["lcf"] * 90)
        # ========== CoPO Modification Ends ==========

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
    from newcopo.metadrive_scenario.marl_envs.marl_waymo_env import MARLWaymoEnv
    from scenarionet_training.marl.utils.env_wrappers import get_lcf_env, get_rllib_compatible_env

    parser = get_train_parser()
    args = parser.parse_args()
    stop = {"timesteps_total": 200_0000}
    exp_name = "test_mappo" if not args.exp_name else args.exp_name
    config = dict(
        env=get_rllib_compatible_env(get_lcf_env(MARLWaymoEnv)),
        env_config=dict(
            discrete_action=True,
            discrete_action_dim=7,
            # start_seed=tune.grid_search([5000]),
            # num_agents=8,
        ),
        # num_sgd_iter=1,
        # rollout_fragment_length=200,
        # train_batch_size=400,
        # sgd_minibatch_size=200,
        num_workers=0,
        # **{USE_CENTRALIZED_CRITIC: True},
        # fuse_mode=tune.grid_search(["concat", "mf"])
        # fuse_mode=tune.grid_search(["none"])

        train_batch_size=100,
        rollout_fragment_length=20,
        sgd_minibatch_size=30,

        discriminator_l2=1e-5,
        discriminator_lr=1e-4,  # They use LR linear decay! We should change this too!
        discriminator_num_iters=3,
        discriminator_sgd_minibatch_size=30,

        enable_discriminator=True,
        discriminator_add_action=False,
        discriminator_reward_native=True,


        counterfactual=False,

        use_copo_dataset=False,
        use_copo_dataset_with_inverse_model=True
    )
    results = train(
        MultiAgentAIRL,
        config=config,  # Do not use validate_config_add_multiagent here!
        checkpoint_freq=0,  # Don't save checkpoint is set to 0.
        keep_checkpoints_num=0,
        stop=stop,
        num_gpus=args.num_gpus,
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
    _test()

    # load_human_data()

