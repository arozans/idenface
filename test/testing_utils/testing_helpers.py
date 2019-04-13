from typing import Tuple

import numpy as np
import tensorflow as tf

from src.utils import configuration, consts
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


def generate_fake_images(size: Tuple[int, ...], mimic_values=None):
    fake_random_images = np.random.uniform(size=size).astype(np.float32)
    if mimic_values is not None:
        for idx, label in enumerate(mimic_values):
            fake_random_images[idx][0] = label / 10
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
