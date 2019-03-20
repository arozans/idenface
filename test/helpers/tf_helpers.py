import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe


def run_eagerly(func):
    @functools.wraps(func)
    def eager_fun(*args, **kwargs):
        with tf.Session() as sess:
            sess.run(tfe.py_func(func, inp=list(kwargs.values()), Tout=[]))

    return eager_fun


def get_first_batch(dataset: tf.data.Dataset):
    iterator = dataset.make_one_shot_iterator()
    first_batch = iterator.get_next()
    with tf.Session() as sess:
        res = sess.run(first_batch)
    return res
