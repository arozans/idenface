import collections
from builtins import KeyError
from typing import Optional, List, Tuple, Dict

import numpy as np
from numpy.core.multiarray import ndarray

from src.data.common_types import DatasetSpec
from src.data.processing import generating_pairs
from src.data.raw_data import raw_data
from src.utils import consts, utils
from src.utils.configuration import config


def create_paired_data(dataset_spec: DatasetSpec) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    raw_images, raw_labels = raw_data.get_raw_data(dataset_spec)

    return _create_paired_data(examples=raw_images, labels=raw_labels, dataset_spec=dataset_spec)


def _create_paired_data(examples: np.ndarray, labels: np.ndarray, dataset_spec: DatasetSpec,
                        size: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if dataset_spec.with_excludes:
        keys_to_drop = []
    else:
        keys_to_drop = config[consts.EXCLUDED_KEYS]

    zipped = zip(examples, labels)
    features_dict = collections.defaultdict(list)
    for x, y in zipped:
        features_dict[y].append(x)

    utils.log("Creating paired data excluding keys: " + str(keys_to_drop))
    try:
        [features_dict.pop(key) for key in keys_to_drop]
    except KeyError as e:
        utils.log("Key to exclude not found in dataset: {}".format(e))

    if size:
        pairs_num = size // 2
    else:
        pairs_num = len(examples) // 2
    same_pairs: List[Tuple[ndarray, ndarray]]
    same_pairs, same_labels = generating_pairs.create_same_pairs(features_dict, pairs_num,
                                                                 dataset_spec)
    diff_pairs: List[Tuple[ndarray, ndarray]]
    left_labels: List[int]
    right_labels: List[int]
    diff_pairs, left_labels, right_labels = generating_pairs.create_different_pairs(features_dict, pairs_num)

    diff_one_hot_labels = [[0]] * len(diff_pairs)
    same_one_hot_labels = [[1]] * len(same_pairs)

    all_pairs: ndarray = np.vstack((same_pairs, diff_pairs))
    left_digit_labels: ndarray = np.hstack((same_labels, left_labels))
    right_digit_labels: ndarray = np.hstack((same_labels, right_labels))
    pair_labels: ndarray = np.vstack((same_one_hot_labels, diff_one_hot_labels))

    all_pairs, pair_labels, all_left_labels, all_right_labels = unison_shuffle(all_pairs, pair_labels,
                                                                               left_digit_labels,
                                                                               right_digit_labels)

    left_pairs, right_pairs = zip(*all_pairs)

    features_dict = collections.OrderedDict(
        {consts.LEFT_FEATURE_IMAGE: np.array(left_pairs), consts.RIGHT_FEATURE_IMAGE: np.array(right_pairs)})
    labels_dict = collections.OrderedDict(
        {consts.LEFT_FEATURE_LABEL: np.array(all_left_labels), consts.RIGHT_FEATURE_LABEL: np.array(all_right_labels),
         consts.PAIR_LABEL: np.array(pair_labels.sum(axis=1))})

    return features_dict, labels_dict


def unison_shuffle(a: ndarray, b: ndarray, c: ndarray, d: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    assert len(a) == len(b) == len(c) == len(d)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p]
