from pathlib import Path
from typing import Dict

import numpy as np
import skimage
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordWriter

from src.data.common_types import DatasetSpec
from src.utils import utils, consts


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _save_pairs_as_raw_bytes(features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], path):
    rows, cols, depth = _validate_paired_data_shape(features, labels)

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
            _save_pair_example(writer, left_raw, right_raw, pair_label, left_label, right_label, cols, rows,
                               depth)


def _save_pairs_encoding(features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], path):
    rows, cols, depth = _validate_paired_data_shape(features, labels)

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
                _save_pair_example(writer, left_encoded, right_encoded, pair_label, left_label, right_label, cols, rows,
                                   depth)


def _save_unpaired_as_raw_bytes(features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], path):
    rows, cols, depth = _validate_unpaired_data_shape(features, labels)

    with tf.python_io.TFRecordWriter(str(path)) as writer:
        for idx, (image, label) in enumerate(
                zip(features[consts.FEATURES], labels[consts.LABELS])):
            image_float = skimage.img_as_float(image)
            image_raw = image_float.tostring()
            _save_unpaired_example(writer, image_raw, label, cols, rows, depth)


def _save_unpaired_encoding(features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], path):
    rows, cols, depth = _validate_unpaired_data_shape(features, labels)

    decoded_image = tf.placeholder(tf.uint16)
    encoding_image = tf.image.encode_png(decoded_image)

    with tf.Session() as sess:
        with TFRecordWriter(str(path)) as writer:
            for idx, (image, label) in enumerate(zip(features[consts.FEATURES], labels[consts.LABELS])):
                if idx % 5000 == 0:
                    print("Encoding sample no: ", idx)

                image_to_encode = skimage.img_as_uint(image)

                image_encoded = sess.run(
                    encoding_image,
                    feed_dict={
                        decoded_image: image_to_encode,
                    })
                _save_unpaired_example(writer, image_encoded, label, cols, rows, depth)


def _save_unpaired_example(writer: TFRecordWriter, image_bytes: tf.Tensor, label: int, cols: int, rows: int,
                           depth: int):
    example = tf.train.Example(features=tf.train.Features(feature={
        consts.TFRECORD_IMAGE_BYTES: _bytes_feature(image_bytes),
        consts.TFRECORD_LABEL: _int64_feature(label),
        consts.TFRECORD_HEIGHT: _int64_feature(rows),
        consts.TFRECORD_WEIGHT: _int64_feature(cols),
        consts.TFRECORD_DEPTH: _int64_feature(depth)
    }))
    writer.write(example.SerializeToString())


def _save_pair_example(writer: TFRecordWriter, left_bytes: tf.Tensor, right_bytes: tf.Tensor, pair_label: int,
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


def _validate_paired_data_shape(features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray]):
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


def _validate_unpaired_data_shape(features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray]):
    images = features[consts.FEATURES]
    labels = labels[consts.LABELS]

    if images.shape[0] != len(labels):
        raise ValueError(
            'Images size {} does not match labels size {}.'.format(images.shape[0], len(labels)))

    image = images[0]
    return image.shape[0], image.shape[1], image.shape[2]


def save_to_tfrecord(data_dict: Dict[str, np.ndarray], data_labels: Dict[str, np.ndarray], path: Path,
                     dataset_spec: DatasetSpec):
    utils.log('Saving .tfrecord file: {} using spec: {}'.format(path, dataset_spec))
    path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_spec.encoding:
        if not dataset_spec.paired:
            _save_unpaired_as_raw_bytes(data_dict, data_labels, path)
        else:
            _save_pairs_as_raw_bytes(data_dict, data_labels, path)
    else:
        if dataset_spec.paired:
            _save_pairs_encoding(data_dict, data_labels, path)
        else:
            _save_unpaired_encoding(data_dict, data_labels, path)
