import torch
from torch import nn
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from scenarionet_training.marl.algo.copo import CoPOModel

class MultiheadDynamicsModel(nn.Module):
    def __init__(self, K, output_dim):
        super(MultiheadDynamicsModel, self).__init__()
        self.K = K

        dynamics_list = []
        for _ in range(K):
            dynamics_parameters = torch.nn.Parameter(
                torch.cat([torch.zeros([output_dim]), torch.zeros([output_dim]) - 2], dim=0),
                requires_grad=True
            )
            dynamics_list.append(dynamics_parameters)
        self.dynamics_list = nn.ModuleList(dynamics_list)

    def forward(self, index):
        assert 0 <= index < self.K
        dynamics_parameters = self.dynamics_list[index]
        loc, log_std = torch.chunk(dynamics_parameters, 2, dim=0)
        std = torch.exp(log_std.clamp(-20, 10))
        dist = torch.distributions.Normal(loc, std)
        dynamics_parameters = dist.rsample()
        return dynamics_parameters, dist


class MultiheadPolicyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, num_multihead):
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name
        )

        self.K = num_multihead

        model_list = []

        for _ in range(self.K):
            model_list.append(
                CoPOModel(
                    obs_space=obs_space,
                    action_space=action_space,
                    num_outputs=num_outputs,
                    model_config=model_config,
                    name=name
                )
            )

        self.model_list = nn.ModuleList(model_list)

ModelCatalog.register_custom_model("multihead_policy_model", MultiheadPolicyModel)
