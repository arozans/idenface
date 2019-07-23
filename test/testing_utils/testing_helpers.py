from pathlib import Path

import imageio
import numpy as np
import tensorflow as tf

from src.data.common_types import RawDatasetFragment, DictsDataset
from src.utils import configuration, consts, filenames


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
        tf.compat.v1.app.run(training.main)
    except SystemExit:
        print("Test main finished")


def save_arrays_as_images_on_disc(fake_random_images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from testing_utils import gen
    image_filenames = []
    filename = filenames.get_raw_input_data_dir()
    if labels is None:
        images_per_label = 2
        labels = range(0, len(fake_random_images), images_per_label)
        labels = sorted((list(labels) * images_per_label))
    for idx, (label, image) in enumerate(zip(labels, fake_random_images)):
        path = (Path(filename) / ("000" + str(label)) / str(idx)).with_suffix(consts.PNG)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            while True:
                unique_path = path.with_name(path.stem + '_' + gen.random_str(3)).with_suffix(path.suffix)
                if not unique_path.exists():
                    path = unique_path
                    break
        image_filenames.append(path)
        imageio.imwrite(path, image)
    return np.array(image_filenames)


def determine_optimizer(optimizer_param):
    if optimizer_param == consts.GRADIENT_DESCEND_OPTIMIZER:
        return tf.train.GradientDescentOptimizer
    elif optimizer_param == consts.ADAM_OPTIMIZER:
        return tf.train.AdamOptimizer
    else:
        raise ValueError("Unknown optimizer: {}".format(optimizer_param))


def unpack_images_dict(images_dict):
    if len(images_dict.values()) == 1:
        features = images_dict[consts.FEATURES]
    else:
        left_batch = images_dict[consts.LEFT_FEATURE_IMAGE]
        right_batch = images_dict[consts.RIGHT_FEATURE_IMAGE]
        features = np.concatenate((left_batch, right_batch))
    return features


def unpack_labels_dict(labels_dict):
    if len(labels_dict.values()) == 1:
        labels = labels_dict[consts.LABELS]
    else:
        left_labels = labels_dict[consts.LEFT_FEATURE_LABEL]
        right_labels = labels_dict[consts.RIGHT_FEATURE_LABEL]
        labels = np.concatenate((left_labels, right_labels))
    return labels


def dicts_dataset_to_raw_dataset_fragment(images_dataset: DictsDataset):
    features = unpack_images_dict(images_dataset.features)
    labels = unpack_labels_dict(images_dataset.labels)

    return RawDatasetFragment(features=features, labels=labels)


def save_images_dict_on_disc(images_dict, labels_dict) -> DictsDataset:
    path_dict_dataset = {}
    if len(images_dict.values()) == 1:
        images = images_dict[consts.FEATURES]
        labels = labels_dict[consts.LABELS]

        paths = save_arrays_as_images_on_disc(images, labels)
        path_dict_dataset = {consts.FEATURES: paths}
    else:
        left_batch = images_dict[consts.LEFT_FEATURE_IMAGE]
        left_labels = labels_dict[consts.LEFT_FEATURE_LABEL]
        paths_left = save_arrays_as_images_on_disc(left_batch, left_labels)
        path_dict_dataset.update({consts.LEFT_FEATURE_IMAGE: paths_left})

        right_batch = images_dict[consts.RIGHT_FEATURE_IMAGE]
        right_labels = labels_dict[consts.RIGHT_FEATURE_LABEL]
        paths_right = save_arrays_as_images_on_disc(right_batch, right_labels)
        path_dict_dataset.update({consts.RIGHT_FEATURE_IMAGE: paths_right})
    return DictsDataset(path_dict_dataset, labels_dict)


def get_full_class_name(clazz):
    module = clazz.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return clazz.__class__.__name__
    return module + '.' + clazz.__class__.__name__
