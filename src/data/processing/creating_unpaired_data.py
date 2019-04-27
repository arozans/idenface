import collections
from typing import Tuple, Dict

import numpy as np
from numpy.core.multiarray import ndarray

from src.data.common_types import DatasetSpec
from src.data.raw_data import raw_data
from src.utils import consts, utils
from src.utils.configuration import config


def create_unpaired_data(dataset_spec: DatasetSpec) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    raw_images, raw_labels = raw_data.get_raw_data(dataset_spec)

    return _create_unpaired_data(examples=raw_images, labels=raw_labels, dataset_spec=dataset_spec)


def _create_unpaired_data(examples: np.ndarray, labels: np.ndarray, dataset_spec: DatasetSpec) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if dataset_spec.with_excludes:
        keys_to_drop = []
    else:
        keys_to_drop = config[consts.EXCLUDED_KEYS]

    utils.log("Creating unpaired data excluding keys: " + str(keys_to_drop))
    examples = np.array(examples)
    labels = np.array(labels)
    if keys_to_drop:
        indexes = np.logical_and.reduce([labels != x for x in keys_to_drop])
        examples = examples[indexes]
        labels = labels[indexes]
    examples, labels = unison_shuffle(examples, labels)
    features_dict = collections.OrderedDict({consts.FEATURES: np.array(examples)})
    labels_dict = collections.OrderedDict({consts.LABELS: np.array(labels)})

    return features_dict, labels_dict


def unison_shuffle(a: ndarray, b: ndarray) -> Tuple[ndarray, ndarray]:
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
