"""
CoPO Implementation:

1. when processing trajectory, we should compute the neighborhood reward and many advantages
2. the local coordination is achieved by replacing the advantage used in PPO loss
3. the global coordination is an independent step in outer PPO loop

PZH
"""
import logging
from collections import defaultdict
from typing import Type

import gym
import numpy as np
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step, multi_gpu_train_one_step
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
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

from scenarionet_training.marl.algo.ccppo import CCModel, CCPPOTrainer, CCPPOPolicy, CCPPOConfig

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


class CoPOConfig(CCPPOConfig):

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


# ========== CoPO Model ==========
class CoPOModel(CCModel):
    """Generic fully connected network implemented in ModelV2 API."""

    def __init__(
            self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
            model_config: ModelConfigDict, name: str
    ):
        super(CoPOModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [])

        # Note: We compute the centralized critic obs size here!
        centralized_critic_obs_dim = self.get_centralized_critic_obs_dim()

        # Build neighbours value function
        self.nei_value_network = self.build_one_value_network(
            in_size=centralized_critic_obs_dim, hiddens=hiddens, activation=activation
        )

        # Build global value function
        self.global_value_network = self.build_one_value_network(
            in_size=centralized_critic_obs_dim, hiddens=hiddens, activation=activation
        )

        if self.model_config["custom_model_config"][USE_DISTRIBUTIONAL_LCF]:
            lcf_parameters = [0.0, np.log(self.model_config["custom_model_config"]["initial_lcf_std"])]
        else:
            lcf_parameters = [0.0]
        self.lcf_parameters = torch.nn.Parameter(torch.as_tensor(lcf_parameters), requires_grad=True)

        self.view_requirements[NEI_REWARDS] = ViewRequirement()
        self.view_requirements[NEI_VALUES] = ViewRequirement()
        self.view_requirements[NEI_TARGET] = ViewRequirement()
        self.view_requirements[NEI_ADVANTAGE] = ViewRequirement()
        self.view_requirements[GLOBAL_REWARDS] = ViewRequirement()
        self.view_requirements[GLOBAL_VALUES] = ViewRequirement()
        self.view_requirements[GLOBAL_TARGET] = ViewRequirement()
        self.view_requirements[GLOBAL_ADVANTAGES] = ViewRequirement()
        self.view_requirements["normalized_advantages"] = ViewRequirement()
        if self.model_config["custom_model_config"][USE_DISTRIBUTIONAL_LCF]:
            self.view_requirements["step_lcf"] = ViewRequirement()

    def build_one_value_network(self, in_size, activation, hiddens):
        assert in_size > 0
        vf_layers = []
        for size in hiddens:
            vf_layers.append(
                SlimFC(
                    in_size=in_size,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(1.0)
                )
            )
            in_size = size
        vf_layers.append(SlimFC(
            in_size=in_size, out_size=1, initializer=normc_initializer(0.01), activation_fn=None
        ))
        return nn.Sequential(*vf_layers)

    def get_nei_value(self, centralized_critic_obs):
        return self._post(self.nei_value_network(centralized_critic_obs))

    def get_global_value(self, centralized_critic_obs):
        return self._post(self.global_value_network(centralized_critic_obs))

    def compute_coordinated(self, ego, neighbor):
        if self.model_config["custom_model_config"][USE_DISTRIBUTIONAL_LCF]:
            # Conduct reparameterization trick here!
            lcf_rad = self.lcf_dist.rsample(ego.size()) * np.pi / 2
        else:
            lcf_rad = self.lcf_mean * np.pi / 2
        return torch.cos(lcf_rad) * ego + torch.sin(lcf_rad) * neighbor

    @property
    def lcf_dist(self):
        if self.model_config["custom_model_config"][USE_DISTRIBUTIONAL_LCF]:
            return torch.distributions.normal.Normal(self.lcf_mean, self.lcf_std)
        else:
            return None

    @property
    def lcf_mean(self):
        return torch.clamp(torch.tanh(self.lcf_parameters[0]), -1 + 1e-6, 1 - 1e-6)

    @property
    def lcf_std(self):
        if self.model_config["custom_model_config"][USE_DISTRIBUTIONAL_LCF]:
            return torch.exp(torch.clamp(self.lcf_parameters[1], -20, 2))
        else:
            return None

    def _post(self, tensor):
        return torch.reshape(tensor, [-1])


ModelCatalog.register_custom_model("copo_model", CoPOModel)


# ========== CoPO Policy ==========
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


class CoPOPolicy(CCPPOPolicy):
    def __init__(self, observation_space, action_space, config):
        super(CoPOPolicy, self).__init__(observation_space, action_space, config)

        # Add a target model
        dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"], framework="torch")
        self.target_model = ModelCatalog.get_model_v2(
            obs_space=observation_space,
            action_space=action_space,
            num_outputs=logit_dim,
            model_config=config["model"],
            framework="torch",
            name="copo_target_model"
        )
        self.target_model.to(self.device)
        self.update_old_policy()  # Note that we are not sure if the model is synced with local policy at this time

        # Setup the LCF optimizer and relevant variables.
        # Note that this optimizer is only used in local policy, but it is defined in all policies.
        self._lcf_optimizer = torch.optim.Adam([self.model.lcf_parameters], lr=self.config[LCF_LR])
        self.view_requirements[SampleBatch.NEXT_OBS] = ViewRequirement(
            SampleBatch.OBS,
            space=observation_space,
            shift=1
        )  # Force to add "new_obs" in the sample batch.

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

    def extra_grad_info(self, train_batch):
        ret = super(CoPOPolicy, self).extra_grad_info(train_batch)

        additional_ret = {}
        for k in ["lcf", "mean_nei_vf_loss", "mean_global_vf_loss", "normalized_advantages"]:
            s = self.get_tower_stats(k)
            additional_ret[k] = torch.mean(torch.stack(s))

        if self.config[USE_DISTRIBUTIONAL_LCF]:
            additional_ret["lcf_std"] = torch.mean(torch.stack(self.get_tower_stats("lcf_std")))

        additional_ret = convert_to_numpy(additional_ret)
        ret.update(additional_ret)
        assert "entropy" in ret, ret.keys()
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

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        sample_batch = super(CoPOPolicy, self).postprocess_trajectory(sample_batch, other_agent_batches, episode)
        with torch.no_grad():
            cobs = convert_to_torch_tensor(sample_batch[CENTRALIZED_CRITIC_OBS], self.device)
            sample_batch[NEI_VALUES] = self.model.get_nei_value(cobs).cpu().detach().numpy().astype(np.float32)
            sample_batch[GLOBAL_VALUES] = self.model.get_global_value(cobs).cpu().detach().numpy().astype(np.float32)

            infos = sample_batch.get(SampleBatch.INFOS)
            if episode is not None:  # After initialization
                assert isinstance(infos[0], dict)
                # Modified: when initialized, add neighborhood/global reward/value
                sample_batch[NEI_REWARDS] = np.array([info[NEI_REWARDS] for info in infos]).astype(np.float32)
                sample_batch[GLOBAL_REWARDS] = np.array([info[GLOBAL_REWARDS] for info in infos]).astype(np.float32)
                if self.config[USE_DISTRIBUTIONAL_LCF]:
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
        assert sample_batch["step_lcf"].max() == sample_batch["step_lcf"].min()
        return sample_batch


# ========== The CoPO Trainer ==========
class CoPOTrainer(CCPPOTrainer):
    @classmethod
    def get_default_config(cls):
        return CoPOConfig()

    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        assert config["framework"] == "torch"
        return CoPOPolicy

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

        # ========== CoPO Modification ==========
        # Standardize advantages
        # train_batch = standardize_fields(train_batch, ["advantages"])  # Native PPO
        local_policy = self.get_policy("default")
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

            # CoPO: Just put the mean and std of advantage in local policy, will be used in Meta Update.
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
        train_results["default"]["custom_metrics"]["meta_update"] = meta_update_fetches
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
    from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
    from scenarionet_training.marl.utils.env_wrappers import get_lcf_env, get_rllib_compatible_env

    parser = get_train_parser()
    args = parser.parse_args()
    stop = {"timesteps_total": 200_0000}
    exp_name = "test_mappo" if not args.exp_name else args.exp_name
    config = dict(
        env=get_rllib_compatible_env(get_lcf_env(MultiAgentIntersectionEnv)),
        env_config=dict(
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
        sgd_minibatch_size=30
    )
    results = train(
        CoPOTrainer,
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
        local_mode=True
    )


if __name__ == "__main__":
    _test()
