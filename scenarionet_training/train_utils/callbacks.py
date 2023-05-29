from typing import Dict

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy


class DrivingCallbacks(DefaultCallbacks):
    def on_episode_start(
            self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
            env_index: int, **kwargs
    ):
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["step_reward"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["lateral_dist"] = []
        episode.user_data["cost"] = []
        episode.user_data["num_crash_vehicle"] = []
        episode.user_data["num_crash_human"] = []
        episode.user_data["num_crash_object"] = []
        episode.user_data["num_on_line"] = []

        episode.user_data["step_reward_lateral"] = []
        episode.user_data["step_reward_heading"] = []
        episode.user_data["step_reward_action_smooth"] = []

    def on_episode_step(
            self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["velocity"].append(info["velocity"])
            episode.user_data["steering"].append(info["steering"])
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["acceleration"].append(info["acceleration"])
            episode.user_data["lateral_dist"].append(info["lateral_dist"])
            episode.user_data["cost"].append(info["cost"])
            for x in ["num_crash_vehicle", "num_crash_object", "num_crash_human", "num_on_line"]:
                episode.user_data[x].append(info[x])

            for x in ["step_reward_lateral", "step_reward_heading", "step_reward_action_smooth"]:
                episode.user_data[x].append(info[x])

    def on_episode_end(
            self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
            **kwargs
    ):
        arrive_dest = episode.last_info_for()["arrive_dest"]
        crash = episode.last_info_for()["crash"]
        out_of_road = episode.last_info_for()["out_of_road"]
        max_step_rate = not (arrive_dest or crash or out_of_road)
        episode.custom_metrics["success_rate"] = float(arrive_dest)
        episode.custom_metrics["crash_rate"] = float(crash)
        episode.custom_metrics["out_of_road_rate"] = float(out_of_road)
        episode.custom_metrics["max_step_rate"] = float(max_step_rate)
        episode.custom_metrics["velocity_max"] = float(np.max(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_mean"] = float(np.mean(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_min"] = float(np.min(episode.user_data["velocity"]))

        episode.custom_metrics["lateral_dist_min"] = float(np.min(episode.user_data["lateral_dist"]))
        episode.custom_metrics["lateral_dist_max"] = float(np.max(episode.user_data["lateral_dist"]))
        episode.custom_metrics["lateral_dist_mean"] = float(np.mean(episode.user_data["lateral_dist"]))

        episode.custom_metrics["steering_max"] = float(np.max(episode.user_data["steering"]))
        episode.custom_metrics["steering_mean"] = float(np.mean(episode.user_data["steering"]))
        episode.custom_metrics["steering_min"] = float(np.min(episode.user_data["steering"]))
        episode.custom_metrics["acceleration_min"] = float(np.min(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_mean"] = float(np.mean(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_max"] = float(np.max(episode.user_data["acceleration"]))
        episode.custom_metrics["step_reward_max"] = float(np.max(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_mean"] = float(np.mean(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_min"] = float(np.min(episode.user_data["step_reward"]))

        episode.custom_metrics["cost"] = float(sum(episode.user_data["cost"]))
        for x in ["num_crash_vehicle", "num_crash_object", "num_crash_human", "num_on_line"]:
            episode.custom_metrics[x] = float(sum(episode.user_data[x]))

        for x in ["step_reward_lateral", "step_reward_heading", "step_reward_action_smooth"]:
            episode.custom_metrics[x] = float(np.mean(episode.user_data[x]))

        episode.custom_metrics["route_completion"] = float(episode.last_info_for()["route_completion"])
        episode.custom_metrics["curriculum_level"] = int(episode.last_info_for()["curriculum_level"])
        episode.custom_metrics["scenario_index"] = int(episode.last_info_for()["scenario_index"])
        episode.custom_metrics["track_length"] = float(episode.last_info_for()["track_length"])
        episode.custom_metrics["num_stored_maps"] = int(episode.last_info_for()["num_stored_maps"])
        episode.custom_metrics["scenario_difficulty"] = float(episode.last_info_for()["scenario_difficulty"])
        episode.custom_metrics["data_coverage"] = float(episode.last_info_for()["data_coverage"])
        episode.custom_metrics["curriculum_success"] = float(episode.last_info_for()["curriculum_success"])
        episode.custom_metrics["curriculum_route_completion"] = float(
            episode.last_info_for()["curriculum_route_completion"])

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["success"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["level"] = np.nan
        result["length"] = result["episode_len_mean"]
        result["coverage"] = np.nan
        if "custom_metrics" not in result:
            return

        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
            result["level"] = result["custom_metrics"]["curriculum_level_mean"]
            result["coverage"] = result["custom_metrics"]["data_coverage_mean"]
