from typing import Dict
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy


class CostCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        episode.user_data["cost"] = []

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        info = episode._last_infos["agent0"]
        if info is not None:
            episode.user_data["cost"].append(info["cost"])

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        episode.custom_metrics["cost"] = float(sum(episode.user_data["cost"]))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        if "cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["cost_mean"]


class DrivingCallbacks(DefaultCallbacks):
    def on_episode_start(
            self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: EpisodeV2,
            env_index: int, **kwargs
    ):
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["step_reward"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["cost"] = []

    def on_episode_step(
            self, *, worker: RolloutWorker, base_env: BaseEnv, episode: EpisodeV2, env_index: int, **kwargs
    ):
        info = episode._last_infos["agent0"]
        if info is not None:
            episode.user_data["velocity"].append(info["velocity"])
            episode.user_data["steering"].append(info["steering"])
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["acceleration"].append(info["acceleration"])
            episode.user_data["cost"].append(info["cost"])

    def on_episode_end(
            self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: EpisodeV2,
            **kwargs
    ):
        arrive_dest = episode._last_infos["agent0"]["arrive_dest"]
        crash = episode._last_infos["agent0"]["crash"]
        out_of_road = episode._last_infos["agent0"]["out_of_road"]
        max_step_rate = not (arrive_dest or crash or out_of_road)
        episode.custom_metrics["success_rate"] = float(arrive_dest)
        episode.custom_metrics["crash_rate"] = float(crash)
        episode.custom_metrics["out_of_road_rate"] = float(out_of_road)
        episode.custom_metrics["max_step_rate"] = float(max_step_rate)
        episode.custom_metrics["velocity_max"] = float(np.max(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_mean"] = float(np.mean(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_min"] = float(np.min(episode.user_data["velocity"]))
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
        episode.custom_metrics["route_completion"] = episode._last_infos["agent0"]["route_completion"]
        episode.custom_metrics["curriculum_level"] = episode._last_infos["agent0"]["curriculum_level"]
        episode.custom_metrics["curriculum_level"] = episode._last_infos["agent0"]["scenario_index"]
        episode.custom_metrics["track_length"] = episode._last_infos["agent0"]["track_length"]

    def on_train_result(self,
                        *,
                        algorithm: "Algorithm",
                        result: dict,
                        **kwargs, ):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["length"] = result["episode_len_mean"]
        result["cost"] = np.nan
        if "custom_metrics" not in result:
            return

        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
        if "cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["cost_mean"]
