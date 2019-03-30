import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from src.utils import consts


def run_eagerly(func):
    @functools.wraps(func)
    def eager_fun(*args, **kwargs):
        with tf.Session() as sess:
            sess.run(tfe.py_func(func, inp=list(kwargs.values()), Tout=[]))

    return eager_fun


def get_first_batch(dataset: tf.data.Dataset):
    iterator = dataset.make_one_shot_iterator()
    first_batch = iterator.get_next()

    return first_batch


def unpack_batch(batch):
    features = batch[0]
    labels = batch[1]
    left_batch = features[consts.LEFT_FEATURE_IMAGE]
    right_batch = features[consts.RIGHT_FEATURE_IMAGE]
    if tf.executing_eagerly():
        left_images = left_batch.numpy()
        right_images = right_batch.numpy()
        labels = labels.numpy()
    else:
        with tf.Session() as sess:
            left_images, right_images, labels = sess.run([left_batch, right_batch, labels])
    return left_images, right_images, labels


def unpack_first_batch(dataset: tf.data.Dataset):
    batch = get_first_batch(dataset)
    return unpack_batch(batch)


def get_string(maybe_str_tensor):
    if tf.executing_eagerly():
        return maybe_str_tensor.numpy().decode("utf-8")
    else:
        return maybe_str_tensor
