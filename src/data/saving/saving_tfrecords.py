from pathlib import Path
from typing import Dict, Union

import numpy as np
import skimage
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordWriter

from src.utils import utils, consts


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _save_as_raw_bytes(features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], path):
    rows, cols, depth = _validate_data_shape(features, labels)

    with tf.python_io.TFRecordWriter(str(path)) as writer:
        for idx, (left, right, pair_label, left_label, right_label) in \
                enumerate(zip(*list(features.values()),
                              labels[consts.PAIR_LABEL],
                              labels[consts.LEFT_FEATURE_LABEL],
                              labels[consts.RIGHT_FEATURE_LABEL]
                              )):
            left_float = skimage.img_as_float(left)
            right_float = skimage.img_as_float(right)
            left_raw = left_float.tostring()
            right_raw = right_float.tostring()
            _save_example(writer, left_raw, right_raw, pair_label, left_label, right_label, cols, rows,
                          depth)


def _save_encoding(features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], path):
    rows, cols, depth = _validate_data_shape(features, labels)

    decoded_left_image = tf.placeholder(tf.uint16)
    decoded_right_image = tf.placeholder(tf.uint16)
    encoding_left = tf.image.encode_png(decoded_left_image)
    encoding_right = tf.image.encode_png(decoded_right_image)

    with tf.Session() as sess:
        with TFRecordWriter(str(path)) as writer:
            for idx, (left, right, pair_label, left_label, right_label) in \
                    enumerate(zip(*list(features.values()),
                                  labels[consts.PAIR_LABEL],
                                  labels[consts.LEFT_FEATURE_LABEL],
                                  labels[consts.RIGHT_FEATURE_LABEL]
                                  )):
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
                _save_example(writer, left_encoded, right_encoded, pair_label, left_label, right_label, cols, rows,
                              depth)


def _save_example(writer: TFRecordWriter, left_bytes: tf.Tensor, right_bytes: tf.Tensor, pair_label: int,
                  left_label: int, right_label: int, cols: int, rows: int, depth: int):
    example = tf.train.Example(features=tf.train.Features(feature={
        consts.TFRECORD_LEFT_BYTES: _bytes_feature(left_bytes),
        consts.TFRECORD_RIGHT_BYTES: _bytes_feature(right_bytes),
        consts.TFRECORD_PAIR_LABEL: _int64_feature(int(pair_label)),
        consts.TFRECORD_LEFT_LABEL: _int64_feature(int(left_label)),
        consts.TFRECORD_RIGHT_LABEL: _int64_feature(int(right_label)),
        consts.TFRECORD_HEIGHT: _int64_feature(rows),
        consts.TFRECORD_WEIGHT: _int64_feature(cols),
        consts.TFRECORD_DEPTH: _int64_feature(depth)
    }))
    writer.write(example.SerializeToString())


def _validate_data_shape(features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray]):
    left_images = np.array(list(features.values())[0])
    right_images = np.array(list(features.values())[1])
    left_labels = labels[consts.LEFT_FEATURE_LABEL]
    right_labels = labels[consts.RIGHT_FEATURE_LABEL]
    same_labels = labels[consts.PAIR_LABEL]

    if left_images.shape[0] != right_images.shape[0] != len(left_labels) != len(right_labels) != len(same_labels):
        raise ValueError('Images size {} and {} does not match labels size {}, {}, {}.'.format(left_images.shape[0],
                                                                                               right_images.shape[0],
                                                                                               len(left_labels),
                                                                                               len(right_labels),
                                                                                               len(same_labels)))
    if left_images.shape != right_images.shape:
        raise ValueError('Left and right images have different shapes!')

    left = left_images[0]
    return left.shape[0], left.shape[1], left.shape[2]


def save_to_tfrecord(data_dict: Dict[str, np.ndarray], data_labels: Dict[str, np.ndarray], path: Path,
                     encoding: Union[None, bool] = True):
    utils.log('Saving .tfrecord file: {}, encoding: {}'.format(path, encoding))
    path.parent.mkdir(parents=True, exist_ok=True)

    if not encoding:
        _save_as_raw_bytes(data_dict, data_labels, path)
    else:
        _save_encoding(data_dict, data_labels, path)
