import os.path as osp

import numpy as np

# root = ~/.../copo/copo/maour_environment
root = osp.dirname(osp.abspath(osp.dirname(__file__)))

_checkpoints_buffers = {}

# There data is generate in compress_trained_model.py
meta_svo_lookup_table = {
    "copo_round_0": (0.3837417275236364, 0.10217650927472532),
    "copo_round_1": (0.31903679224482756, 0.0923634324418871),
    # "copo_round_1": (0.31903679224482756, 0.0923634324418871),  ###  0.856
    "copo_round_2": (0.4292658473315993, 0.09482618003037936),
    "copo_round_3": (0.4448881541427814, 0.10107655640234027),
    "copo_round_4": (0.40086105256749877, 0.09221974747222766),
    "copo_parking_0": (0.30009075371842636, 0.09819950084937246),
    "copo_parking_1": (0.21065708838011088, 0.09828158781716699),
    "copo_parking": (0.21065708838011088, 0.09828158781716699),  # Best
    "copo_parking_2": (0.19518211745379263, 0.099467324583154),
    "copo_parking_3": (0.10191127059883193, 0.0997653921183787),
    "copo_parking_4": (0.16749037122517296, 0.10430529321494854),
    "copo_bottle_0": (0.3347182310464089, 0.09320298072538878),
    "copo_bottle_1": (0.17889355489036493, 0.09873832422390318),
    "copo_bottle_2": (0.20677767223433444, 0.09703644548068967),
    "copo_bottle": (0.20677767223433444, 0.09703644548068967),  # 0.867, Best
    "copo_bottle_3": (0.38850163995173936, 0.0996062973873657),
    "copo_bottle_4": (0.41495788567944586, 0.09026645110887394),
    "copo_inter_0": (0.36824979071031544, 0.08807231132921418),
    "copo_inter": (0.36824979071031544, 0.08807231132921418),  # Best
    "copo_inter_1": (0.3538261389261798, 0.0960544714410054),
    "copo_inter_2": (0.5021972039289642, 0.09395808752691537),
    "copo_inter_3": (0.32071430693592934, 0.09482878145941516),
    "copo_inter_4": (0.5012396887729041, 0.08545188030652832),
    "copo_round_rerun_0": (0.18783088442112683, 0.09685282814254507),
    "copo_round_rerun_1": (0.4449950145496117, 0.08596959420113016),
    "copo_round_rerun_2": (0.2914212175433245, 0.09590505765930911),
    "copo_round": (0.2914212175433245, 0.09590505765930911),  # 0.858, Best
    "copo_round_rerun_3": (0.3506030522549751, 0.09272900488746863),
    "copo_bottle_rerun_0": (0.21729068847457367, 0.09800391086381884),
    "copo_bottle_rerun_1": (0.31267254543763706, 0.0914876350830348),
    # "copo_bottle_rerun_1": (0.31267254543763706, 0.0914876350830348),  ### 0.839
    "copo_bottle_rerun_2": (0.20579787078985448, 0.09402470028045275),
    "copo_tollgate_0": (0.46550068926742755, 0.08945204678064445),
    "copo_tollgate_1": (0.4772816712447233, 0.08097108654084174),
    "copo_tollgate_2": (0.4913835221499055, 0.08520848447553676),
    "copo_tollgate_3": (0.5575323092877565, 0.07595817525083297),
    "copo_tollgate": (0.5575323092877565, 0.07595817525083297),  # Best
    "copo_tollgate_4": (0.5247444219924696, 0.08257146898526042),
}


def _compute_actions_for_tf_policy(
        weights, obs, deterministic=False, policy_name="default_policy", layer_name_suffix="", return_details=None
):
    obs = np.asarray(obs)
    assert obs.ndim == 2
    s = "{}/fc_1{}/kernel".format(policy_name, layer_name_suffix)
    assert s in weights, (s, weights.keys())
    assert obs.shape[1] == weights[s].shape[0], (obs.shape, weights[s].shape)
    x = np.matmul(
        obs,
        weights["{}/fc_1{}/kernel".format(policy_name, layer_name_suffix)]) + \
        weights["{}/fc_1{}/bias".format(policy_name, layer_name_suffix)]
    x = np.tanh(x)
    x = np.matmul(
        x,
        weights["{}/fc_2{}/kernel".format(policy_name, layer_name_suffix)]) + \
        weights["{}/fc_2{}/bias".format(policy_name, layer_name_suffix)]
    x = np.tanh(x)
    x = np.matmul(
        x,
        weights["{}/fc_out{}/kernel".format(policy_name, layer_name_suffix)]) + \
        weights["{}/fc_out{}/bias".format(policy_name, layer_name_suffix)]
    # x = x.reshape(-1)
    mean, log_std = np.split(x, 2, axis=1)
    if deterministic:
        return mean
    std = np.exp(log_std)
    action = np.random.normal(mean, std)
    return action


def _compute_actions_for_torch_policy(weights, obs, deterministic=False, policy_name=None, layer_name_suffix=None,
                                      return_details=False):
    obs = np.asarray(obs)
    assert obs.ndim == 2
    x = np.matmul(obs, weights["_hidden_layers.0._model.0.weight"].T) + weights["_hidden_layers.0._model.0.bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["_hidden_layers.1._model.0.weight"].T) + weights["_hidden_layers.1._model.0.bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["_logits._model.0.weight"].T) + weights["_logits._model.0.bias"]
    mean, log_std = np.split(x, 2, axis=1)
    if deterministic:
        if return_details:
            return mean, {"mean": mean, "avg_mean": np.mean(mean)}
        return mean
    std = np.exp(log_std)
    action = np.random.normal(mean, std)
    if return_details:
        return action, {"mean": mean, "std": std, "avg_mean": np.mean(mean), "avg_std": np.mean(std)}
    return action


def get_policy_function(model_name: str, checkpoint_dir_name="checkpoints"):
    # checkpoint_name is like: {ALGO}_{ENV}_{INDEX}.npz
    global _checkpoints_buffers
    if model_name not in _checkpoints_buffers:
        path = osp.join(root, checkpoint_dir_name, model_name + ".npz")
        w = np.load(path)
        w = {k: w[k] for k in w.files}
        _checkpoints_buffers[model_name] = w
    else:
        w = _checkpoints_buffers[model_name]

    if model_name.startswith("ccppo"):
        return lambda obs: _compute_actions_for_torch_policy(w, obs)
    elif model_name.startswith("ippo"):
        return lambda obs: _compute_actions_for_tf_policy(w, obs, policy_name="default", layer_name_suffix="")
    elif model_name.startswith("cl"):
        return lambda obs: _compute_actions_for_tf_policy(w, obs, policy_name="default", layer_name_suffix="")
    elif model_name.startswith("copo"):
        return lambda obs: _compute_actions_for_tf_policy(w, obs, policy_name="default", layer_name_suffix="_1")
    else:
        raise ValueError("Unknown model: ", model_name)


class PolicyFunction:
    def __init__(
            self,
            model_name,
            use_distributional_svo=True,
            auto_add_svo_to_obs=True,
            checkpoint_dir_name="best_checkpoints"
    ):
        self.policy = get_policy_function(model_name, checkpoint_dir_name)

        self.model_name = model_name
        self.use_svo = model_name.startswith("copo")
        self.existing_svo = dict()
        self.use_distributional_svo = use_distributional_svo
        self.auto_add_svo_to_obs = auto_add_svo_to_obs

    def __call__(self, obs_dict, last_done_dict):
        obs_dict = self.process_svo(obs_dict)
        obs_to_be_eval = []
        obs_to_be_eval_keys = []
        for agent_id, agent_ob in obs_dict.items():  # I don't know why there is one 'agent0' extra here!
            if (agent_id not in last_done_dict) or (not last_done_dict.get(agent_id, False)):
                obs_to_be_eval.append(agent_ob)
                obs_to_be_eval_keys.append(agent_id)
        actions = self.policy(obs_to_be_eval)
        action_to_send = {}
        for count, agent_id in enumerate(obs_to_be_eval_keys):
            action_to_send[agent_id] = actions[count]
        return action_to_send

    def process_svo(self, obs_dict):
        if (not self.use_svo) or (not self.auto_add_svo_to_obs):
            return obs_dict

        new_dict = {}
        for k, o in obs_dict.items():
            if k not in self.existing_svo:
                svo_mean, svo_std = meta_svo_lookup_table[self.model_name]
                if self.use_distributional_svo:
                    self.existing_svo[k] = np.clip(np.random.normal(loc=svo_mean, scale=svo_std), -1, 1)
                else:
                    self.existing_svo[k] = svo_mean
                # print("Current SVO mean is {}, std {}. We choose {}".format(svo_mean, svo_std, self.existing_svo[k]))
            # o = np.concatenate([o, [self.existing_svo[k]]])
            chosen_svo = (self.existing_svo[k] + 1) / 2
            o = np.concatenate([o, [chosen_svo]])
            new_dict[k] = o
        return new_dict

    def reset(self):
        self.existing_svo.clear()


if __name__ == '__main__':
    for env, shape in {"round": 91, "inter": 91, "parking": 91, "bottle": 96, "tollgate": 156}.items():
        for algo in [
            "copo",
        ]:
            checkpoint_name = "{}_{}_0".format(algo, env)
            print("Start running: ", checkpoint_name)
            # f = get_policy_function(checkpoint_name)
            f = PolicyFunction(checkpoint_name, use_distributional_svo=True if "meta" in algo else False)

            for _ in range(5):
                r = f(
                    {"agent{}".format(i): np.zeros(shape=[
                        shape,
                    ])
                        for i in range(10)}, {"agent{}".format(i): False
                                              for i in range(10)}
                )
            # print(r.shape)
            print(r.keys())
