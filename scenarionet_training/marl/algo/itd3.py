import gym
from ray.rllib.algorithms.td3 import TD3, TD3Config
from ray.rllib.policy.policy import PolicySpec


class ITD3Config(TD3Config):

    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or ITD3)

        self.hidden_size = 256
        self.hidden_layer = 2

        self.framework_str = "torch"

        self.critic_lr = 1e-4
        self.actor_lr = 1e-4

    def validate(self):
        self.actor_hiddens = [self.hidden_size] * self.hidden_layer
        self.critic_hiddens = [self.hidden_size] * self.hidden_layer

        # Note that in new RLLib the rollout_fragment_length will auto adjust to a new value
        # so that one pass of all workers and envs will collect enough data for a train batch.
        super().validate()

        from ray.tune.registry import _global_registry, ENV_CREATOR
        from metadrive.constants import DEFAULT_AGENT

        env_class = _global_registry.get(ENV_CREATOR, self["env"])
        single_env = env_class(self["env_config"])

        if "agent0" in single_env.observation_space.spaces:
            obs_space = single_env.observation_space["agent0"]
            act_space = single_env.action_space["agent0"]
        else:
            obs_space = single_env.observation_space[DEFAULT_AGENT]
            act_space = single_env.action_space[DEFAULT_AGENT]

        assert isinstance(obs_space, gym.spaces.Box)
        # assert isinstance(act_space, gym.spaces.Box)
        # Note that we can't set policy name to "default_policy" since by doing so
        # ray will record wrong per agent episode reward!
        self.update_from_dict({"multiagent":
            dict(
                # Note that we have to use "default" because stupid RLLib has bug when
                # we are using "default_policy" as the Policy ID.
                policies={"default": PolicySpec(None, obs_space, act_space, {})},
                policy_mapping_fn=lambda x: "default"
            )
        })

        # single_env.close()


class ITD3(TD3):
    @classmethod
    def get_default_config(cls):
        return ITD3Config()
