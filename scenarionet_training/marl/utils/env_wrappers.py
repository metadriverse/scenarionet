import copy
from collections import defaultdict
from math import cos, sin

import numpy as np
from gym.spaces import Box, Dict
from metadrive.utils import get_np_random, clip
from ray.rllib.env import MultiAgentEnv
from ray.tune.registry import register_env
import os.path as osp
from collections import defaultdict, deque

NEWCOPO_DIR = osp.dirname(osp.dirname(osp.abspath(osp.dirname(__file__))))

COMM_ACTIONS = "comm_actions"
COMM_PREV_ACTIONS = "comm_prev_actions"

# prev_obs_{t-1} is the concatenation of neighbors' message comm_action_{t-1}
COMM_PREV_OBS = "comm_prev_obs"

# current_obs_t is the concatenation of neighbors' message comm_action_{t-1}
COMM_CURRENT_OBS = "comm_current_obs"
COMM_PREV_2_OBS = "comm_prev_2_obs"

COMM_LOGITS = "comm_logits"
COMM_LOG_PROB = "comm_log_prob"
ENV_PREV_OBS = "env_prev_obs"

COMM_METHOD = "comm_method"

NEI_OBS = "nei_obs"


def get_raw_state(self):
    ret = dict(position=self.position, heading=self.heading, velocity=self.velocity)
    return ret

class CCEnv:
    """
    This class maintains a distance map of all agents and appends the
    neighbours' names and distances into info at each step.
    We should subclass this class to a base environment class.
    """

    @classmethod
    def default_config(cls):
        config = super(CCEnv, cls).default_config()
        # Note that this config is set to 40 in LCFEnv
        config["neighbours_distance"] = 40

        config.update(dict(
            communication=dict(
                comm_method="none",
                comm_size=4,
                comm_neighbours=4,
                add_pos_in_comm=False
            ),
            add_traffic_light=False,
            traffic_light_interval=30,
        ))

        return config

    def __init__(self, *args, **kwargs):
        super(CCEnv, self).__init__(*args, **kwargs)
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))
        if self.config["communication"][COMM_METHOD] != "none":
            self._comm_obs_buffer = defaultdict()

        if self.config["communication"]["add_pos_in_comm"]:
            self._comm_dim = self.config["communication"]["comm_size"] + 3
        else:
            self._comm_dim = self.config["communication"]["comm_size"]

    def _get_reset_return(self):
        if self.config["communication"][COMM_METHOD] != "none":
            self._comm_obs_buffer = defaultdict()
        return super(CCEnv, self)._get_reset_return()

    @property
    def action_space(self):
        old_action_space = super(CCEnv, self).action_space
        if not self.config["communication"][COMM_METHOD] != "none":
            return old_action_space
        assert isinstance(old_action_space, Dict)
        new_action_space = Dict(
            {k: Box(
                low=single.low[0],
                high=single.high[0],
                dtype=single.dtype,

                # We are not using self._comm_dim here!
                shape=(single.shape[0] + self.config["communication"]["comm_size"],)
            )
                for k, single in old_action_space.spaces.items()
            }
        )
        return new_action_space

    def step(self, actions):

        if self.config["communication"][COMM_METHOD] != "none":
            comm_actions = {k: v[2:] for k, v in actions.items()}
            actions = {k: v[:2] for k, v in actions.items()}

        o, r, d, i = super(CCEnv, self).step(actions)
        self._update_distance_map(dones=d)
        for kkk in i.keys():
            i[kkk]["all_agents"] = list(i.keys())

            neighbours, nei_distances = self._find_in_range(kkk, self.config["neighbours_distance"])
            i[kkk]["neighbours"] = neighbours
            i[kkk]["neighbours_distance"] = nei_distances

            if self.config["communication"][COMM_METHOD] != "none":
                i[kkk][COMM_CURRENT_OBS] = []
                for n in neighbours[: self.config["communication"]["comm_neighbours"]]:
                    if n in comm_actions:
                        if self.config["communication"]["add_pos_in_comm"]:
                            ego_vehicle = self.vehicles_including_just_terminated[kkk]
                            nei_vehicle = self.vehicles_including_just_terminated[n]
                            relative_position = ego_vehicle.projection(nei_vehicle.position - ego_vehicle.position)
                            dis = np.linalg.norm(relative_position)
                            extra_comm_obs = [
                                dis / 20,
                                ((relative_position[0] / dis) + 1) / 2,
                                ((relative_position[1] / dis) + 1) / 2
                            ]
                            tmp_comm_obs = np.concatenate([comm_actions[n], np.clip(np.asarray(extra_comm_obs), 0, 1)])
                        else:
                            tmp_comm_obs = comm_actions[n]
                        i[kkk][COMM_CURRENT_OBS].append(tmp_comm_obs)
                    else:
                        i[kkk][COMM_CURRENT_OBS].append(np.zeros((self._comm_dim,)))

        return o, r, d, i

    def _find_in_range(self, v_id, distance):
        if distance <= 0:
            return [], []
        max_distance = distance
        dist_to_others = self.distance_map[v_id]
        dist_to_others_list = sorted(dist_to_others, key=lambda k: dist_to_others[k])
        ret = [
            dist_to_others_list[i] for i in range(len(dist_to_others_list))
            if dist_to_others[dist_to_others_list[i]] < max_distance
        ]
        ret2 = [
            dist_to_others[dist_to_others_list[i]] for i in range(len(dist_to_others_list))
            if dist_to_others[dist_to_others_list[i]] < max_distance
        ]
        return ret, ret2

    def _update_distance_map(self, dones=None):
        self.distance_map.clear()
        if hasattr(self, "vehicles_including_just_terminated"):
            vehicles = self.vehicles_including_just_terminated
            # if dones is not None:
            #     assert (set(dones.keys()) - set(["__all__"])) == set(vehicles.keys()), (dones, vehicles)
        else:
            vehicles = self.vehicles  # Fallback to old version MetaDrive, but this is not accurate!
        keys = [k for k, v in vehicles.items() if v is not None]
        for c1 in range(0, len(keys) - 1):
            for c2 in range(c1 + 1, len(keys)):
                k1 = keys[c1]
                k2 = keys[c2]
                p1 = vehicles[k1].position
                p2 = vehicles[k2].position
                distance = np.linalg.norm(p1 - p2)
                self.distance_map[k1][k2] = distance
                self.distance_map[k2][k1] = distance


class LCFEnv(CCEnv):
    @classmethod
    def default_config(cls):
        config = super(LCFEnv, cls).default_config()
        config.update(
            dict(
                # Overwrite the CCEnv's neighbours_distance=10 to 40.
                neighbours_distance=40,

                # Two mode to compute utility for each vehicle:
                # "linear": util = r_me * lcf + r_other * (1 - lcf), lcf in [0, 1]
                # "angle": util = r_me * cos(lcf) + r_other * sin(lcf), lcf in [0, pi/2]
                # "angle" seems to be more stable!
                lcf_mode="angle",
                lcf_dist="normal",  # "uniform" or "normal"
                lcf_normal_std=0.1,  # The initial STD of normal distribution, might change by calling functions.

                # If this is set to False, then the return reward is natively the LCF-weighted coordinated reward!
                # This will be helpful in ablation study!
                return_native_reward=True,

                # Whether to force set the lcf
                force_lcf=-100,

                enable_copo=True
            )
        )
        return config

    def __init__(self, config=None):
        super(LCFEnv, self).__init__(config)
        self.lcf_map = {}
        assert hasattr(super(LCFEnv, self), "_update_distance_map")
        assert self.config["lcf_mode"] in ["linear", "angle"]
        assert self.config["lcf_dist"] in ["uniform", "normal"]
        assert self.config["lcf_normal_std"] > 0.0
        self.force_lcf = self.config["force_lcf"]

        # Only used in normal LCF distribution
        # LCF is always in range [0, 1], but the real LCF degree is in [-pi/2, pi/2].
        self.current_lcf_mean = 0.0  # Set to 0 degree.
        self.current_lcf_std = self.config["lcf_normal_std"]

        self._last_obs = None
        self._traffic_light_counter = 0

    @property
    def enable_copo(self):
        return self.config["enable_copo"]

    def get_single_observation(self, vehicle_config):
        original_obs = super(LCFEnv, self).get_single_observation(vehicle_config)

        if not self.enable_copo:
            return original_obs

        original_obs_cls = original_obs.__class__
        original_obs_name = original_obs_cls.__name__
        comm_method = self.config["communication"][COMM_METHOD]

        single_comm_dim = self.config["communication"]["comm_size"]
        if self.config["communication"]["add_pos_in_comm"]:
            single_comm_dim += 3
        comm_obs_size = single_comm_dim * self.config["communication"]["comm_neighbours"]

        add_traffic_light = self.config["add_traffic_light"]

        class LCFObs(original_obs_cls):
            @property
            def observation_space(self):
                space = super(LCFObs, self).observation_space
                assert isinstance(space, Box)
                assert len(space.shape) == 1
                length = space.shape[0] + 1

                if comm_method != "none":
                    length += comm_obs_size

                if add_traffic_light:
                    length += 1 + 2  # Global position should be put

                # Note that original metadrive obs space is [0, 1]
                space = Box(
                    low=np.array([-1.0] * length),
                    high=np.array([1.0] * length),
                    shape=(length,),
                    dtype=space.dtype
                )
                space._shape = space.shape
                return space

        LCFObs.__name__ = original_obs_name
        LCFObs.__qualname__ = original_obs_name

        # TODO: This part is not beautiful! Refactor in future release!
        # from metadrive.envs.marl_envs.tinyinter import CommunicationObservation
        # if original_obs_cls == CommunicationObservation:
        #     return LCFObs(vehicle_config, self)
        # else:
        return LCFObs(vehicle_config)

    @property
    def _traffic_light_msg(self):
        fix_interval = self.config["traffic_light_interval"]
        increment = (self._traffic_light_counter % fix_interval) / fix_interval * 0.1
        if ((self._traffic_light_counter // fix_interval) % 2) == 1:
            return 0 + increment
        else:
            return 1 - increment

    def get_agent_traffic_light_msg(self, pos):
        b_box = self.engine.current_map.road_network.get_bounding_box()
        pos0 = (pos[0] - b_box[0]) / (b_box[1] - b_box[0])
        pos1 = (pos[1] - b_box[2]) / (b_box[3] - b_box[2])
        # print("Msg: {}, Pos0: {}, Pos1 {}".format(self._traffic_light_msg, pos0, pos1))
        return np.clip(np.array([self._traffic_light_msg, pos0, pos1]), 0, 1).astype(np.float32)

    def _get_reset_return(self):
        self.lcf_map.clear()
        self._update_distance_map()
        obses = super(LCFEnv, self)._get_reset_return()

        if self.config["add_traffic_light"]:
            self._traffic_light_counter = 0
            new_obses = {}
            for agent_name, v in self.vehicles_including_just_terminated.items():
                if agent_name not in obses:
                    continue
                new_obses[agent_name] = np.concatenate([
                    obses[agent_name],
                    self.get_agent_traffic_light_msg(v.position)
                ])
            obses = new_obses

        ret = {}
        for k, o in obses.items():
            lcf, ret[k] = self._add_lcf(o)
            self.lcf_map[k] = lcf

        yet_another_new_obs = {}
        if self.config["communication"][COMM_METHOD] != "none":
            for k, old_obs in ret.items():
                yet_another_new_obs[k] = np.concatenate(
                    [
                        old_obs,
                        np.zeros((
                            self._comm_dim * self.config["communication"]["comm_neighbours"],
                        ))
                    ], axis=-1
                ).astype(np.float32)

            ret = yet_another_new_obs

        self._last_obs = ret
        return ret

    def step(self, actions):
        # step the environment
        o, r, d, i = super(LCFEnv, self).step(actions)
        assert set(i.keys()) == set(o.keys())
        new_obs = {}
        new_rewards = {}
        global_reward = sum(r.values()) / len(r.values())

        if self.config["add_traffic_light"]:
            self._traffic_light_counter += 1

        for agent_name, agent_info in i.items():
            assert "neighbours" in agent_info
            # Note: agent_info["neighbours"] records the neighbours within radius neighbours_distance.
            nei_rewards = [r[nei_name] for nei_name in agent_info["neighbours"]]
            if nei_rewards:
                i[agent_name]["nei_rewards"] = sum(nei_rewards) / len(nei_rewards)
            else:
                i[agent_name]["nei_rewards"] = 0.0  # Do not provide neighbour rewards if no neighbour
            i[agent_name]["global_rewards"] = global_reward

            if self.config["add_traffic_light"]:
                o[agent_name] = np.concatenate([
                    o[agent_name],
                    self.get_agent_traffic_light_msg(
                        self.vehicles_including_just_terminated[agent_name].position
                    )
                ])

            # add LCF into observation, also update LCF map and info.
            agent_lcf, new_obs[agent_name] = self._add_lcf(
                agent_obs=o[agent_name],
                lcf=self.lcf_map[agent_name] if agent_name in self.lcf_map else None
            )
            if agent_name not in self.lcf_map:
                # The agent LCF is set for the whole episode
                self.lcf_map[agent_name] = agent_lcf
            i[agent_name]["lcf"] = agent_lcf
            i[agent_name]["lcf_deg"] = agent_lcf * 90

            # lcf_map stores values in [-1, 1]
            if self.config["lcf_mode"] == "linear":
                assert 0.0 <= agent_lcf <= 1.0
                new_r = agent_lcf * r[agent_name] + (1 - agent_lcf) * agent_info["nei_rewards"]
            elif self.config["lcf_mode"] == "angle":
                assert -1.0 <= agent_lcf <= 1.0
                lcf_rad = agent_lcf * np.pi / 2
                new_r = cos(lcf_rad) * r[agent_name] + sin(lcf_rad) * agent_info["nei_rewards"]
            else:
                raise ValueError("Unknown LCF mode: {}".format(self.config["lcf_mode"]))
            i[agent_name]["coordinated_rewards"] = new_r
            i[agent_name]["native_rewards"] = r[agent_name]
            if self.config["return_native_reward"]:
                new_rewards[agent_name] = r[agent_name]
            else:
                new_rewards[agent_name] = new_r

        yet_another_new_obs = {}
        if self.config["communication"][COMM_METHOD] != "none":
            for k, old_obs in new_obs.items():
                comm_obs = i[k][COMM_CURRENT_OBS]
                if len(comm_obs) < self.config["communication"]["comm_neighbours"]:
                    comm_obs.extend(
                        [np.zeros((self._comm_dim,))] *
                        (self.config["communication"]["comm_neighbours"] - len(comm_obs))
                    )
                yet_another_new_obs[k] = np.concatenate([old_obs] + comm_obs).astype(np.float32)

            new_obs = yet_another_new_obs

            for kkk in i.keys():
                neighbours = i[kkk]["neighbours"]
                i[kkk]["nei_obs"] = []
                for nei_index in range(self.config["communication"]["comm_neighbours"]):
                    if nei_index >= len(neighbours):
                        n = None
                    else:
                        n = neighbours[nei_index]
                    if n is not None and n in self._last_obs:
                        i[kkk]["nei_obs"].append(self._last_obs[n])
                    else:
                        i[kkk]["nei_obs"].append(None)
                i[kkk]["nei_obs"].append(None)  # Adding extra None to make sure np.array fails!

        self._last_obs = new_obs
        return new_obs, new_rewards, d, i

    def _add_lcf(self, agent_obs, lcf=None):

        if not self.enable_copo:
            return 0.0, agent_obs

        if self.force_lcf != -100:
            # Set LCF to given value
            if self.config["lcf_dist"] == "normal":
                assert -1.0 <= self.force_lcf <= 1.0
                lcf = get_np_random().normal(loc=self.force_lcf, scale=self.current_lcf_std)
                lcf = clip(lcf, -1, 1)
            else:
                lcf = self.force_lcf
        elif lcf is not None:
            pass
        else:
            # Sample LCF value from current distribution
            if self.config["lcf_dist"] == "normal":
                assert -1.0 <= self.current_lcf_mean <= 1.0
                lcf = get_np_random().normal(loc=self.current_lcf_mean, scale=self.current_lcf_std)
                lcf = clip(lcf, -1, 1)
            else:
                lcf = get_np_random().uniform(-1, 1)
        assert -1.0 <= lcf <= 1.0
        output_lcf = (lcf + 1) / 2  # scale to [0, 1]
        return lcf, np.float32(np.concatenate([agent_obs, [output_lcf]]))

    def set_lcf_dist(self, mean, std):
        assert self.enable_copo
        assert self.config["lcf_dist"] == "normal"
        self.current_lcf_mean = mean
        self.current_lcf_std = std
        assert std > 0.0
        assert -1.0 <= self.current_lcf_mean <= 1.0

    def set_force_lcf(self, v):
        assert self.enable_copo
        self.force_lcf = v


def get_ccenv(env_class):
    name = env_class.__name__

    class TMP(CCEnv, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP


def get_change_n_env(env_class):
    class ChangeNEnv(env_class):
        def __init__(self, config):
            self._raw_input_config = copy.deepcopy(config)
            super(ChangeNEnv, self).__init__(config)

        def close_and_reset_num_agents(self, num_agents):
            config = copy.deepcopy(self._raw_input_config)
            self.close()
            config["num_agents"] = num_agents
            super(ChangeNEnv, self).__init__(config)

    name = env_class.__name__
    name = "CL{}".format(name)
    ChangeNEnv.__name__ = name
    ChangeNEnv.__qualname__ = name
    return ChangeNEnv


def get_lcf_env(env_class):
    name = env_class.__name__

    class TMP(LCFEnv, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP



class LatentEnvBase:
    @classmethod
    def default_config(cls):
        config = super(LatentEnvBase, cls).default_config()
        config.update(dict(
            enable_latent=False,
            latent_dim=-1,
            # latent_trajectory_len=-1,  # For LatentPosteriorEnvBase
        ))
        return config

    def __init__(self, config=None):

        super(LatentEnvBase, self).__init__(config)
        # self.latent_map = {}
        self.latent_dict = None

    """
    We have two way to create latent for each independent car:
    The first is we pass a latent_dict[EnvSeed][VehicleName] from external by calling `register_latent` (in this class)
    The second is we can have a function to create latent variable on-the-fly by calling get_latent_for. But this
    method requires a in-environment model registered before rolling out. See LatentPosteriorEnvBase
    """
    def register_latent(self, latent_dict):
        self.latent_dict = latent_dict

    def get_latent_for(self, agent_name):
        if self.latent_dict is None:
            latent = np.zeros(self.config["latent_dim"])
        else:

            if self.engine.global_seed not in self.latent_dict:
                self.latent_dict[self.engine.global_seed] = {}
            if agent_name not in self.latent_dict[self.engine.global_seed]:
                latent = np.random.normal(0, 1, self.config["latent_dim"])

                self.latent_dict[self.engine.global_seed][agent_name] = latent

            # if self.engine.global_seed not in self.latent_dict:
            #     print("latent_dict: {} ({}), seed: {}".format(
            #         self.latent_dict.keys(), len(self.latent_dict), self.engine.global_seed))
            latent = self.latent_dict[self.engine.global_seed][agent_name]

        return latent

    def _add_latent(self, obs, agent_name):
        if not self.config["enable_latent"]:
            return obs

        latent = self.get_latent_for(agent_name)
        return np.concatenate([latent, obs], axis=-1)

    def reset(self):
        # self.latent_map = {}
        obses = super().reset()
        ret = {}
        for k, o in obses.items():
            ret[k] = self._add_latent(o, agent_name=k)
        return ret

    def step(self, action):
        obses, r, d, i = super().step(action)
        ret = {}
        for k, o in obses.items():
            ret[k] = self._add_latent(o, agent_name=k)
        return ret, r, d, i

    def get_single_observation(self, vehicle_config):
        original_obs = super().get_single_observation(vehicle_config)
        if not self.config["enable_latent"]:
            return original_obs
        original_obs_cls = original_obs.__class__
        original_obs_name = original_obs_cls.__name__
        latent_dim = self.config["latent_dim"]

        class NewObs(original_obs_cls):
            @property
            def observation_space(self):
                space = super(NewObs, self).observation_space
                length = space.shape[0] + latent_dim
                space = Box(
                    low=float("-inf"),
                    high=float("+inf"),
                    shape=(length,),
                    dtype=space.dtype
                )
                space._shape = space.shape
                return space

        NewObs.__name__ = original_obs_name
        NewObs.__qualname__ = original_obs_name
        return NewObs(vehicle_config)


class LatentPosteriorEnvBase(LatentEnvBase):
    """
    Compared to LatentEnvBase, we do not require to register a latent_dict from external.
    We require to register a latent posterior model P(latent | expert trajectory) from external and wrap it into a
    function.
    """
    def register_latent(self, latent_dict):
        raise ValueError("Don't call this function for " + str(self.__class__))

    def register_latent_function(self, latent_function):
        """latent_function(expert_traj in (T, obs_dim)) -> latent (1D array with shape (m,))"""
        self.latent_function = latent_function

        self._human_state_buffer = {}

        # Reset latent dict
        self.latent_dict = {}

    def reset(self, *args, **kwargs):
        self._human_state_buffer = {}
        self._carsize_dict = {}
        self._trajectory_dict = defaultdict(lambda: deque(maxlen=20))
        return super().reset(*args, **kwargs)

    def get_latent_for(self, agent_name):
        if self.latent_dict is None:
            latent = np.zeros(self.config["latent_dim"])  # The environment is not fully initialized yet.
        else:
            if self.engine.global_seed not in self.latent_dict:
                self.latent_dict[self.engine.global_seed] = {}
            if agent_name not in self.latent_dict[self.engine.global_seed]:
                # expert_traj = self.engine.agent_manager.get_expert_trajectory(agent_name)
                # assert expert_traj.shape[0] == self.config["latent_trajectory_len"]

                # if not self._human_state_buffer:
                #     import pickle
                #
                #     human_data_dir = osp.join(NEWCOPO_DIR, "2023-02-06_generate_waymo_data/waymo_human_states_0206")
                #     p = osp.join(human_data_dir, "{}.pkl".format(self.engine.global_seed))
                #     if osp.isfile(p):
                #         with open(p, "rb") as f:
                #             data = pickle.load(f)
                #         self._human_state_buffer = data
                #     else:
                #         self._human_state_buffer = {}
                #
                # latent = None
                # ok_time_step = []
                # for time in self._human_state_buffer.keys():
                #     if time + 1 not in self._human_state_buffer:
                #         continue
                #     if agent_name in self._human_state_buffer[time] and agent_name in self._human_state_buffer[time + 1]:
                #         ok_time_step.append(time)
                #
                # if len(ok_time_step) > 0:
                #     time = np.random.choice(ok_time_step)
                #
                #     if agent_name in self.agent_manager._agent_to_object:
                #         carsize = np.array([
                #                 self.agent_manager.get_agent(agent_name).WIDTH / 10,
                #                 self.agent_manager.get_agent(agent_name).LENGTH / 10
                #             ])
                #     else:
                #         carsize = np.array([2.5 / 10, 5 / 10])

                latent = self.latent_function(
                    self.engine.global_seed,
                    agent_name,
                    # self._human_state_buffer[time][agent_name],
                    # self._human_state_buffer[time + 1][agent_name],
                    # carsize
                )

                if latent is None:
                    latent = np.random.normal(0, 1, self.config["latent_dim"])

                self.latent_dict[self.engine.global_seed][agent_name] = latent
            latent = self.latent_dict[self.engine.global_seed][agent_name]
        return latent


def get_latent_env(env_class):
    name = env_class.__name__

    class TMP(LatentEnvBase, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP


def get_latent_posterior_env(env_class):
    name = env_class.__name__

    class TMP(LatentPosteriorEnvBase, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP


def get_rllib_compatible_env(env_class, return_class=False):
    env_name = env_class.__name__

    class MA(env_class, MultiAgentEnv):
        _agent_ids = ["agent{}".format(i) for i in range(100)] + ["{}".format(i) for i in range(10000)] + ["sdc"]

        def __init__(self, *args, **kwargs):
            env_class.__init__(self, *args, **kwargs)
            MultiAgentEnv.__init__(self)

        @property
        def observation_space(self):
            ret = super(MA, self).observation_space
            if not hasattr(ret, "keys"):
                ret.keys = ret.spaces.keys
            return ret

        @property
        def action_space(self):
            ret = super(MA, self).action_space
            if not hasattr(ret, "keys"):
                ret.keys = ret.spaces.keys
            return ret

        def action_space_sample(self, agent_ids: list = None):
            """
            RLLib always has unnecessary stupid requirements that you have to bypass them by overwriting some
            useless functions.
            """
            return self.action_space.sample()

    MA.__name__ = env_name
    MA.__qualname__ = env_name
    register_env(env_name, lambda config: MA(config))

    if return_class:
        return env_name, MA

    return env_name


if __name__ == '__main__':
    # Test if the distance map is correctly updated.
    # from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
    from newcopo.metadrive_scenario.marl_envs.marl_waymo_env import MARLWaymoEnv
    from tqdm import trange

    env = get_latent_posterior_env(get_lcf_env(MARLWaymoEnv))({})
    env.reset()

    for _ in trange(10):
        for _ in trange(1000):
            o, r, d, i = env.step({k: [0, 1] for k in env.vehicles})
            # print(d)
            print(env.vehicles_including_just_terminated)
            if d["__all__"]:
                env.reset()
                break

    env.close()
