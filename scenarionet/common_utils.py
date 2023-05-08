import pickle

import numpy as np
from metadrive.scenario import utils as sd_utils


def recursive_equal(data1, data2, need_assert=False):
    from metadrive.utils.config import Config
    if isinstance(data1, Config):
        data1 = data1.get_dict()
    if isinstance(data2, Config):
        data2 = data2.get_dict()

    if isinstance(data1, np.ndarray):
        tmp = np.asarray(data2)
        return np.all(data1 == tmp)

    if isinstance(data2, np.ndarray):
        tmp = np.asarray(data1)
        return np.all(tmp == data2)

    if isinstance(data1, dict):
        is_ins = isinstance(data2, dict)
        key_right = set(data1.keys()) == set(data2.keys())
        if need_assert:
            assert is_ins and key_right, (data1.keys(), data2.keys())
        if not (is_ins and key_right):
            return False
        ret = []
        for k in data1:
            ret.append(recursive_equal(data1[k], data2[k], need_assert=need_assert))
        return all(ret)

    elif isinstance(data1, (list, tuple)):
        len_right = len(data1) == len(data2)
        is_ins = isinstance(data2, (list, tuple))
        if need_assert:
            assert len_right and is_ins, (len(data1), len(data2), data1, data2)
        if not (is_ins and len_right):
            return False
        ret = []
        for i in range(len(data1)):
            ret.append(recursive_equal(data1[i], data2[i], need_assert=need_assert))
        return all(ret)
    elif isinstance(data1, np.ndarray):
        ret = np.isclose(data1, data2).all()
        if need_assert:
            assert ret, (type(data1), type(data2), data1, data2)
        return ret
    else:
        ret = data1 == data2
        if need_assert:
            assert ret, (type(data1), type(data2), data1, data2)
        return ret


def dict_recursive_remove_array_and_set(d):
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, set):
        return tuple(d)
    if isinstance(d, dict):
        for k in d.keys():
            d[k] = dict_recursive_remove_array_and_set(d[k])
    return d


def save_summary_anda_mapping(summary_file_path, mapping_file_path, summary, mapping):
    with open(summary_file_path, "wb") as file:
        pickle.dump(dict_recursive_remove_array_and_set(summary), file)
    with open(mapping_file_path, "wb") as file:
        pickle.dump(mapping, file)
    print("Dataset Summary and Mapping are saved at: {}".format(summary_file_path))


def read_dataset_summary(dataset_path):
    return sd_utils.read_dataset_summary(dataset_path)


def read_scenario(pkl_file_path):
    return sd_utils.read_scenario_data(pkl_file_path)
