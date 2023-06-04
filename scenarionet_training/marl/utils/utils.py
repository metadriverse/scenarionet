import argparse
import copy
import datetime
import json
import logging
import numbers
import os
import time
from collections import deque
from collections.abc import Iterable
from typing import Optional

import numpy as np
import ray
from ray import rllib


class ForceFPS:
    UNLIMITED = "UnlimitedFPS"
    FORCED = "ForceFPS"

    def __init__(self, fps, start=False):
        self.init_fps = fps
        if start:
            print("We will force the FPS to be near {}".format(fps))
            self.state = self.FORCED
            self.fps = fps + 1  # If we set to 10, FPS will jump in 9~10.
            self.interval = 1 / self.fps
        else:
            self.state = self.UNLIMITED
            self.fps = None
            self.interval = None
        self.dt_stack = deque(maxlen=10)
        self.last_time = time.time()

    def clear(self):
        self.dt_stack.clear()
        self.last_time = time.time()

    def sleep_if_needed(self):
        if self.fps is None:
            return
        self.dt_stack.append(time.time() - self.last_time)
        average_dt = sum(self.dt_stack) / len(self.dt_stack)
        if (self.interval - average_dt) > 0:
            time.sleep(self.interval - average_dt)
        self.last_time = time.time()


def merge_dicts(d1, d2):
    """
    Args:
        d1 (dict): Dict 1.
        d2 (dict): Dict 2.

    Returns:
         dict: A new dict that is d1 and d2 deep merged.
    """
    merged = copy.deepcopy(d1)
    deep_update(merged, d2, True, [])
    return merged


def deep_update(original,
                new_dict,
                new_keys_allowed=False,
                allow_new_subkey_list=None,
                override_all_if_type_changes=None):
    """Updates original dict with values from new_dict recursively.

    If new key is introduced in new_dict, then if new_keys_allowed is not
    True, an error will be thrown. Further, for sub-dicts, if the key is
    in the allow_new_subkey_list, then new subkeys can be introduced.

    Args:
        original (dict): Dictionary with default values.
        new_dict (dict): Dictionary with values to be updated
        new_keys_allowed (bool): Whether new keys are allowed.
        allow_new_subkey_list (Optional[List[str]]): List of keys that
            correspond to dict values where new subkeys can be introduced.
            This is only at the top level.
        override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (dict), iff the "type" key in that value dict changes.
    """
    allow_new_subkey_list = allow_new_subkey_list or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise Exception("Unknown config parameter `{}` ".format(k))

        # Both orginal value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Allowed key -> ok to add new subkeys.
            elif k in allow_new_subkey_list:
                deep_update(original[k], value, True)
            # Non-allowed key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M")


def same_padding(in_size, filter_size, stride_size):
    """
    PZH: Copied from RLLib.

    Note: Padding is added to match TF conv2d `same` padding. See
    www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution

    Args:
        in_size (tuple): Rows (Height), Column (Width) for input
        stride_size (Union[int,Tuple[int, int]]): Rows (Height), column (Width)
            for stride. If int, height == width.
        filter_size (tuple): Rows (Height), column (Width) for filter

    Returns:
        padding (tuple): For input into torch.nn.ZeroPad2d.
        output (tuple): Output shape after padding and convolution.
    """
    in_height, in_width = in_size
    if isinstance(filter_size, int):
        filter_height, filter_width = filter_size, filter_size
    else:
        filter_height, filter_width = filter_size

    stride_size = stride_size if isinstance(stride_size, Iterable) else [stride_size, stride_size]
    stride_height, stride_width = stride_size

    out_height = np.ceil(float(in_height) / float(stride_height))
    out_width = np.ceil(float(in_width) / float(stride_width))

    pad_along_height = int(
        ((out_height - 1) * stride_height + filter_height - in_height))
    pad_along_width = int(
        ((out_width - 1) * stride_width + filter_width - in_width))
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    output = (out_height, out_width)
    return padding, output


class SafeJSONEncoder(json.JSONEncoder):
    def __init__(self, nan_str="null", **kwargs):
        super(SafeJSONEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def default(self, value):
        try:
            if (type(value).__module__ == np.__name__
                    and isinstance(value, np.ndarray)):
                return value.tolist()

            if isinstance(value, np.bool_):
                return bool(value)

            if np.isnan(value):
                return self.nan_str

            if issubclass(type(value), numbers.Integral):
                return int(value)
            if issubclass(type(value), numbers.Number):
                return float(value)

            return super(SafeJSONEncoder, self).default(value)

        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def initialize_ray(local_mode=False, num_gpus=None, test_mode=False, **kwargs):
    os.environ['OMP_NUM_THREADS'] = '1'

    if ray.__version__.split(".")[0] == "1":  # 1.0 version Ray
        if "redis_password" in kwargs:
            redis_password = kwargs.pop("redis_password")
            kwargs["_redis_password"] = redis_password

    ray.init(
        logging_level=logging.ERROR if not test_mode else logging.DEBUG,
        log_to_driver=test_mode,
        local_mode=local_mode,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
        include_dashboard=False,
        **kwargs
    )
    print("Successfully initialize Ray!")
    try:
        print("Available resources: ", ray.available_resources())
    except Exception:
        pass


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-cpus-per-worker", type=float, default=0.5)
    parser.add_argument("--num-gpus-per-trial", type=float, default=0.25)
    parser.add_argument("--test", action="store_true")
    return parser


def setup_logger(debug=False):
    import logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )


def local_test_rllib_check():
    from metadrive.envs.marl_envs import MultiAgentTollgateEnv, MultiAgentIntersectionEnv
    env = get_rllib_compatible_env(MultiAgentTollgateEnv, return_class=True)[1]()
    env.reset()
    rllib.utils.check_env(env)
    env.close()

    env = get_rllib_compatible_env(MultiAgentIntersectionEnv, return_class=True)[1]()
    env.reset()
    rllib.utils.check_env(env)
    env.close()


if __name__ == '__main__':
    local_test_rllib_check()
