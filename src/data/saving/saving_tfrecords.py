from pathlib import Path
from typing import Dict, Union

import numpy as np
import skimage
import tensorflow as tf

from src.utils import utils


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _save_as_raw_bytes(features: Dict[str, np.ndarray], labels: np.ndarray, path):
    rows, cols, depth = _validate_data_shape(features, labels)

    with tf.python_io.TFRecordWriter(str(path)) as writer:
        for left, right, label in zip(*list(features.values()), labels):
            left_float = skimage.img_as_float(left)
            right_float = skimage.img_as_float(right)
            left_raw = left_float.tostring()
            right_raw = right_float.tostring()
            _save_example(cols, depth, label, left_raw, right_raw, rows, writer)


def _save_encoding(features: Dict[str, np.ndarray], labels: np.ndarray, path):
    rows, cols, depth = _validate_data_shape(features, labels)

    decoded_left_image = tf.placeholder(tf.uint16)
    decoded_right_image = tf.placeholder(tf.uint16)
    encoding_left = tf.image.encode_png(decoded_left_image)
    encoding_right = tf.image.encode_png(decoded_right_image)

    with tf.Session() as sess:
        with tf.python_io.TFRecordWriter(str(path)) as writer:
            for idx, (left, right, label) in enumerate(zip(*list(features.values()), labels)):
                if idx % 5000 == 0:
                    print("Encoding sample no: ", idx)

                # or tf.image.convert_image_dtype(left_to_encode, tf.uint16)
                left_to_encode = skimage.img_as_uint(left)
                right_to_encode = skimage.img_as_uint(right)

                left_encoded, right_encoded = sess.run(
                    [encoding_left, encoding_right],
                    feed_dict={
                        decoded_left_image: left_to_encode,
                        decoded_right_image: right_to_encode
                    })
                _save_example(cols, depth, label, left_encoded, right_encoded, rows, writer)


def _save_example(cols, depth, label, left_bytes, right_bytes, rows, writer):
    example = tf.train.Example(features=tf.train.Features(feature={
        'left_bytes': _bytes_feature(left_bytes),
        'right_bytes': _bytes_feature(right_bytes),
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


def save_to_tfrecord(data_dict: Dict[str, np.ndarray], data_labels: np.ndarray, path: Path,
                     encoding: Union[None, bool] = True):
    utils.log('Saving .tfrecord file: {}, encoding: {}'.format(path, encoding))
    path.parent.mkdir(parents=True, exist_ok=True)

    if not encoding:
        _save_as_raw_bytes(data_dict, data_labels, path)
    else:
        _save_encoding(data_dict, data_labels, path)
