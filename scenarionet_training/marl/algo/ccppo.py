"""
Our implementation logic is that:

1. when getting the centralized values, the centralized critic should have a varied input size
2. the centralized observation should be computed when processing trajectory
3. centralized critic should be properly called and used.
"""

from typing import Type

import gym
import numpy as np
# import torch
# import torch.nn as nn
from gym.spaces import Box
from ray import tune
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TrainerConfigDict

from scenarionet_training.marl.algo.ippo import IPPOTrainer, IPPOConfig
from scenarionet_training.marl.utils.callbacks import MultiAgentDrivingCallbacks
from scenarionet_training.marl.utils.env_wrappers import get_ccenv, get_rllib_compatible_env
from scenarionet_training.marl.utils.train import train
from scenarionet_training.marl.utils.utils import get_train_parser

torch, nn = try_import_torch()

CENTRALIZED_CRITIC_OBS = "centralized_critic_obs"
COUNTERFACTUAL = "counterfactual"


class CCPPOConfig(IPPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or CCPPOTrainer)
        self.counterfactual = True
        self.num_neighbours = 4
        self.fuse_mode = "mf"  # In ["concat", "mf", "none"]
        self.mf_nei_distance = 10
        self.old_value_loss = True
        self.update_from_dict({
            "model": {"custom_model": "cc_model"}
        })

    def validate(self):
        super().validate()
        assert self["fuse_mode"] in ["mf", "concat", "none"]
        self.model["custom_model_config"]["fuse_mode"] = self["fuse_mode"]
        self.model["custom_model_config"]["counterfactual"] = self["counterfactual"]
        self.model["custom_model_config"]["num_neighbours"] = self["num_neighbours"]


def get_centralized_critic_obs_dim(
        observation_space_shape, action_space_shape, counterfactual, num_neighbours, fuse_mode
):
    """Get the centralized critic"""
    if fuse_mode == "concat":
        pass
    elif fuse_mode == "mf":
        num_neighbours = 1
    elif fuse_mode == "none":  # Do not use centralized critic
        num_neighbours = 0
    else:
        raise ValueError("Unknown fuse mode: ", fuse_mode)
    num_neighbours += 1
    centralized_critic_obs_dim = num_neighbours * observation_space_shape.shape[0]
    if counterfactual:  # Do not include ego action!
        centralized_critic_obs_dim += (num_neighbours - 1) * action_space_shape.shape[0]
    return centralized_critic_obs_dim


class CCModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(
            self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
            model_config: ModelConfigDict, name: str
    ):

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

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

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # ========== Our Modification: We compute the centralized critic obs size here! ==========
        centralized_critic_obs_dim = self.get_centralized_critic_obs_dim()

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation
                )
            )
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation
                )
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation
                    )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.

            # ========== Our Modification ==========
            # Note: We use centralized critic obs size as the input size of critic!
            # prev_vf_layer_size = int(np.product(obs_space.shape))
            prev_vf_layer_size = centralized_critic_obs_dim
            assert prev_vf_layer_size > 0

            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0)
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_vf_layer_size, out_size=1, initializer=normc_initializer(0.01), activation_fn=None
        )

        self.view_requirements[CENTRALIZED_CRITIC_OBS] = ViewRequirement(
            space=Box(obs_space.low[0], obs_space.high[0], shape=(centralized_critic_obs_dim,))
        )

        self.view_requirements[SampleBatch.ACTIONS] = ViewRequirement(space=action_space)

    def get_centralized_critic_obs_dim(self):
        return get_centralized_critic_obs_dim(
            self.obs_space, self.action_space, self.model_config["custom_model_config"]["counterfactual"],
            self.model_config["custom_model_config"]["num_neighbours"],
            self.model_config["custom_model_config"]["fuse_mode"]
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs = obs.reshape(obs.shape[0], -1)
        features = self._hidden_layers(obs)
        logits = self._logits(features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        raise ValueError("Centralized Value Function should not be called directly! "
                         "Call central_value_function(cobs) instead!")

    def central_value_function(self, obs):
        assert self._value_branch is not None
        return torch.reshape(self._value_branch(self._value_branch_separate(obs)), [-1])


ModelCatalog.register_custom_model("cc_model", CCModel)


def concat_ccppo_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
    """Concat the neighbors' observations"""
    for index in range(sample_batch.count):

        environmental_time_step = sample_batch["t"][index]

        neighbours = sample_batch['infos'][index]["neighbours"]

        # Note that neighbours returned by the environment are already sorted based on their
        # distance to the ego vehicle whose info is being used here.
        for nei_count, nei_name in enumerate(neighbours):
            if nei_count >= policy.config["num_neighbours"]:
                break

            nei_act = None
            nei_obs = None
            if nei_name in other_agent_batches:
                _, nei_batch = other_agent_batches[nei_name]

                match_its_step = np.where(nei_batch["t"] == environmental_time_step)[0]

                if len(match_its_step) == 0:
                    pass
                elif len(match_its_step) > 1:
                    raise ValueError()
                else:
                    new_index = match_its_step[0]
                    nei_obs = nei_batch[SampleBatch.CUR_OBS][new_index]
                    nei_act = nei_batch[SampleBatch.ACTIONS][new_index]

            if nei_obs is not None:
                start = odim + nei_count * other_info_dim
                sample_batch[CENTRALIZED_CRITIC_OBS][index, start: start + odim] = nei_obs
                if policy.config[COUNTERFACTUAL]:
                    sample_batch[CENTRALIZED_CRITIC_OBS][index, start + odim: start + odim + adim] = nei_act
                    assert start + odim + adim == start + other_info_dim
                else:
                    assert start + odim == start + other_info_dim
    return sample_batch


def mean_field_ccppo_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
    """Average the neighbors' observations and probably actions."""
    # Note: Average other's observation might not be a good idea.
    # Maybe we can do some feature extraction before averaging all observations

    assert odim + other_info_dim == sample_batch[CENTRALIZED_CRITIC_OBS].shape[1]
    for index in range(sample_batch.count):

        environmental_time_step = sample_batch["t"][index]


        neighbours = sample_batch['infos'][index]["neighbours"]
        neighbours_distance = sample_batch['infos'][index]["neighbours_distance"]

        obs_list = []
        act_list = []

        for nei_count, (nei_name, nei_dist) in enumerate(zip(neighbours, neighbours_distance)):
            if nei_dist > policy.config["mf_nei_distance"]:
                continue

            nei_act = None
            nei_obs = None
            if nei_name in other_agent_batches:
                _, nei_batch = other_agent_batches[nei_name]

                match_its_step = np.where(nei_batch["t"] == environmental_time_step)[0]

                if len(match_its_step) == 0:
                    pass
                elif len(match_its_step) > 1:
                    raise ValueError()
                else:
                    new_index = match_its_step[0]
                    nei_obs = nei_batch[SampleBatch.CUR_OBS][new_index]
                    nei_act = nei_batch[SampleBatch.ACTIONS][new_index]

            if nei_obs is not None:
                obs_list.append(nei_obs)
                act_list.append(nei_act)

        if len(obs_list) > 0:
            sample_batch[CENTRALIZED_CRITIC_OBS][index, odim:odim + odim] = np.mean(obs_list, axis=0)
            if policy.config[COUNTERFACTUAL]:
                sample_batch[CENTRALIZED_CRITIC_OBS][index, odim + odim:odim + odim + adim] = np.mean(act_list, axis=0)

    return sample_batch


def get_ccppo_env(env_class):
    return get_rllib_compatible_env(get_ccenv(env_class))


class CCPPOPolicy(PPOTorchPolicy):
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        return {}

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        with torch.no_grad():
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
            batch = compute_advantages(
                sample_batch,
                last_r,
                self.config["gamma"],
                self.config["lambda"],
                use_gae=self.config["use_gae"],
                use_critic=self.config.get("use_critic", True)
            )
        return batch

    def loss(self, model, dist_class, train_batch):
        """
        Compute loss for Proximal Policy Objective.

        PZH: We replace the value function here so that we query the centralized values instead
        of the native value function.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        # Compute a value function loss.
        assert self.config["use_critic"]

        # ========== Modification ==========
        # value_fn_out = model.value_function()
        value_fn_out = self.model.central_value_function(train_batch[CENTRALIZED_CRITIC_OBS])
        # ========== Modification Ends ==========

        if self.config["old_value_loss"]:
            current_vf = value_fn_out
            prev_vf = train_batch[SampleBatch.VF_PREDS]
            vf_loss1 = torch.pow(current_vf - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_clipped = prev_vf + torch.clamp(
                current_vf - prev_vf, -self.config["vf_clip_param"], self.config["vf_clip_param"]
            )
            vf_loss2 = torch.pow(vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.max(vf_loss1, vf_loss2)
        else:
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss


class CCPPOTrainer(IPPOTrainer):
    @classmethod
    def get_default_config(cls):
        return CCPPOConfig()

    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        assert config["framework"] == "torch"
        return CCPPOPolicy


def _test():
    # Testing only!
    from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv

    parser = get_train_parser()
    args = parser.parse_args()
    stop = {"timesteps_total": 200_0000}
    exp_name = "test_mappo" if not args.exp_name else args.exp_name
    config = dict(
        env=get_ccppo_env(MultiAgentRoundaboutEnv),
        env_config=dict(
            start_seed=tune.grid_search([5000]),
            num_agents=10,
        ),
        num_sgd_iter=1,
        rollout_fragment_length=200,
        train_batch_size=400,
        sgd_minibatch_size=256,
        num_workers=0,
        # **{COUNTERFACTUAL: tune.grid_search([True, False])},
        **{COUNTERFACTUAL: tune.grid_search([True, ])},
        # fuse_mode=tune.grid_search(["concat", "mf"])
        fuse_mode=tune.grid_search(["mf"])
    )
    results = train(
        CCPPOTrainer,
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
