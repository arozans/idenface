from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from src.utils import utils


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _save(features: Dict[str, np.ndarray], labels: np.ndarray, path):
    rows, cols, depth = _validate_data_shape(features, labels)

    with tf.python_io.TFRecordWriter(str(path)) as writer:
        for left, right, label in zip(*list(features.values()), labels):
            left_raw = left.tostring()
            right_raw = right.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'left_raw': _bytes_feature(left_raw),
                'right_raw': _bytes_feature(right_raw),
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(label))}))
            writer.write(example.SerializeToString())


def _validate_data_shape(features: Dict[str, np.ndarray], labels: np.ndarray):
    left_images = np.array(list(features.values())[0])
    right_images = np.array(list(features.values())[1])
    if left_images.shape[0] != right_images.shape[0] != len(
            labels):
        raise ValueError('Images size %d and %d does not match label size %d.' %
                         (left_images.shape[0],
                          right_images.shape[0],
                          len(labels)))
    if left_images.shape != right_images.shape:
        raise ValueError('Left and right images have different shapes!')

    left = left_images[0]
    return left.shape[0], left.shape[1], left.shape[2]


def save_to_tfrecord(data_dict: Dict[str, np.ndarray], data_labels: np.ndarray, path: Path):
    utils.log('Saving .tfrecord file: {}'.format(path))
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    _save(data_dict, data_labels, path)
