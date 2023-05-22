from ray.rllib.utils import check_env
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.envs.gymnasium_wrapper import GymnasiumEnvWrapper
from gym import Env

if __name__ == '__main__':
    env = GymnasiumEnvWrapper.build(ScenarioEnv)()
    print(isinstance(ScenarioEnv, Env))
    print(isinstance(env, Env))
    print(env.observation_space)
    check_env(env)
