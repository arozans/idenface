from pathlib import Path
from typing import Tuple, List

import imageio
import numpy as np
import tensorflow as tf

from src.data.common_types import DatasetStorageMethod
from src.utils import configuration, consts, filenames
from src.utils.configuration import config


class NumberTranslation:
    ONE_TRANS = ("jeden", "uno", "ein")
    TWO_TRANS = ("dwa", "dos", "zwei")
    THREE_TRANS = ("trzy", "tres", "drei")

    def __init__(self, number: int, trans: str):
        self.number = number
        self.trans = trans

    def __repr__(self):
        return "{}({})".format(self.trans, self.number)

    def __hash__(self):
        return hash(self.trans)

    def __eq__(self, other):
        return self.trans == other.trans

    def __ne__(self, other):
        return not self == other


def run_app():
    from src.estimator.training import training
    configuration.define_cli_args()
    try:
        tf.app.run(training.main)
    except(SystemExit):
        print("Test main finished")


def save_arrays_as_images_on_disc(fake_random_images: np.ndarray, labels: np.ndarray) -> List[Path]:
    image_filenames = []
    filename = filenames.get_raw_input_data_dir()
    if labels is None:
        images_per_label = 2
        labels = range(0, len(fake_random_images), images_per_label)
        labels = sorted((list(labels) * images_per_label))
    for idx, (label, image) in enumerate(zip(labels, fake_random_images)):
        path = Path(filename) / ("000" + str(label)) / str(idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = path.with_suffix(consts.PNG)
        image_filenames.append(path)
        imageio.imwrite(path, image)
    return np.array(image_filenames)


def generate_fake_images(size: Tuple[int, ...],
                         storage_method: DatasetStorageMethod = DatasetStorageMethod.IN_MEMORY,
                         mimic_values=None):
    fake_random_images = np.random.uniform(size=size).astype(np.float32)
    if mimic_values is not None:
        for idx, label in enumerate(mimic_values):
            fake_random_images[idx][0] = label / 10
    if storage_method == DatasetStorageMethod.ON_DISC:
        return save_arrays_as_images_on_disc(fake_random_images, mimic_values)
    else:
        return fake_random_images


def generate_fake_labels(size: int, classes=10, curated=False):
    if curated:
        two_elems_of_each_class = list(np.arange(classes)) * 2
        remainder = np.random.randint(low=0, high=classes, size=size - 2 * classes).astype(np.int64)
        return np.concatenate((two_elems_of_each_class, remainder))
    else:
        return np.random.randint(low=0, high=classes, size=size).astype(np.int64)


def set_test_param(key, param):
    config.testing_helpers.set_test_params({key: param})


def non_streaming_accuracy(predictions, labels):
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def determine_optimizer(optimizer_param):
    if optimizer_param == consts.GRADIENT_DESCEND_OPTIMIZER:
        return tf.train.GradientDescentOptimizer
    elif optimizer_param == consts.ADAM_OPTIMIZER:
        return tf.train.AdamOptimizer
    else:
        raise ValueError("Unknown optimizer: {}".format(optimizer_param))


def save_save_dataset_dict_on_disc(images_dataset_dict, labels_dataset_dict):
    if len(images_dataset_dict.values()) == 1:
        images = images_dataset_dict[consts.FEATURES]
        labels = labels_dataset_dict[consts.LABELS]

        paths = save_arrays_as_images_on_disc(images, labels)
        images_dataset_dict.update({consts.FEATURES: paths})
    else:
        left_batch = images_dataset_dict[consts.LEFT_FEATURE_IMAGE]
        left_labels = labels_dataset_dict[consts.LEFT_FEATURE_LABEL]
        paths_left = save_arrays_as_images_on_disc(left_batch, left_labels)
        images_dataset_dict.update({consts.LEFT_FEATURE_IMAGE: paths_left})

        right_batch = images_dataset_dict[consts.RIGHT_FEATURE_IMAGE]
        right_labels = labels_dataset_dict[consts.RIGHT_FEATURE_LABEL]
        paths_right = save_arrays_as_images_on_disc(right_batch, right_labels)
        images_dataset_dict.update({consts.RIGHT_FEATURE_IMAGE: paths_right})
    return images_dataset_dict
    # pair_labels = labels_dataset_dict[consts.PAIR_LABEL]
