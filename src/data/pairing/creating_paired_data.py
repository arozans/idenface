import collections
from builtins import KeyError
from typing import Optional, List, Tuple, Dict

import numpy as np
from numpy.core.multiarray import ndarray

from src.data.common_types import DatasetSpec
from src.data.pairing import generating_pairs
from src.data.raw_data import raw_data
from src.utils import consts, utils
from src.utils.configuration import config


def create_paired_data(dataset_spec: DatasetSpec):
    raw_images, raw_labels = raw_data.get_raw_data(dataset_spec)
    if dataset_spec.with_excludes:
        keys_to_drop = []
    else:
        keys_to_drop = config.excluded_keys
    return _create_paired_data(examples=raw_images, labels=raw_labels, keys_to_drop=keys_to_drop)


def _create_paired_data(examples: np.ndarray, labels: np.ndarray, keys_to_drop: Optional[list] = None,
                        size: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """

    Returns:
    OrderedDict with two values, each is a ndarray with shape (60000,728) and labels - ndarray with shape (60000,1)
    """
    if keys_to_drop is None:
        keys_to_drop = []
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

    same_pairs: List[Tuple[ndarray, ndarray]] = generating_pairs.create_same_pairs(features_dict, pairs_num,
                                                                                   config.pairing_with_identical)
    diff_pairs: List[Tuple[ndarray, ndarray]] = generating_pairs.create_different_pairs(features_dict, pairs_num)

    diff_one_hot_labels = [[0]] * len(diff_pairs)
    same_one_hot_labels = [[1]] * len(same_pairs)

    all_pairs: ndarray = np.vstack((same_pairs, diff_pairs))
    all_labels: ndarray = np.vstack((same_one_hot_labels, diff_one_hot_labels))

    all_pairs, all_labels = unison_shuffle(all_pairs, all_labels)

    left_pairs, right_pairs = zip(*all_pairs)

    features = collections.OrderedDict(
        {consts.LEFT_FEATURE_IMAGE: np.array(left_pairs), consts.RIGHT_FEATURE_IMAGE: np.array(right_pairs)})

    return features, all_labels.sum(axis=1)


def unison_shuffle(a: ndarray, b: ndarray) -> Tuple[ndarray, ndarray]:
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
