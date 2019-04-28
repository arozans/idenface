from abc import ABC, abstractmethod
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


def save_to_tfrecord(data_dict: Dict[str, np.ndarray], data_labels: Dict[str, np.ndarray], path: Path,
                     dataset_spec: DatasetSpec):
    utils.log('Saving .tfrecord file: {} using spec: {}'.format(path, dataset_spec))
    path.parent.mkdir(parents=True, exist_ok=True)
    tf_savers = {
        (False, False): UnpairedNotEncodingTFRecordSaver,
        (True, False): PairedNotEncodingTFRecordSaver,
        (True, True): PairedEncodingTFRecordSaver,
        (False, True): UnpairedEncodingTFRecordSaver
    }
    saver = tf_savers[(dataset_spec.paired, dataset_spec.encoding)]
    saver(dataset_spec).save(data_dict, data_labels, path)


class AbstractTFRecordSaver:
    def __init__(self, dataset_spec: DatasetSpec):
        self.dataset_spec = dataset_spec

    def save(self, features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], path: Path):
        # rows, cols, depth = \
        self.validate_data_shape(features, labels)
        self.resolve_additional_features(features, labels)
        self._save_to_tfrecord(features, labels, path)

    @abstractmethod
    def validate_data_shape(self, features, labels):
        pass

    @abstractmethod
    def _save_to_tfrecord(self, features, labels, path):
        pass

    @abstractmethod
    def get_features_to_process(self, elems):
        pass

    @abstractmethod
    def get_labels(self, elems):
        pass

    def resolve_additional_features(self, features, labels):
        pass

    def add_additional_features(self, features):
        return features

    @abstractmethod
    def save_example_op(self, *args, **kwargs):
        pass

    def _save_example(self, features, writer):
        features = self.add_additional_features(features)
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())


class EncodingTFRecordSaver(ABC, AbstractTFRecordSaver):

    @abstractmethod
    def get_encoding_ops(self):
        pass

    def _save_to_tfrecord(self, features, labels, path):
        encoding_placeholders, encoding_ops = self.get_encoding_ops()
        with tf.Session() as sess:
            with TFRecordWriter(str(path)) as writer:
                for idx, elems in \
                        enumerate(zip(*list(features.values()), *list(labels.values()))):
                    features_to_bytes = self.get_features_to_process(elems)
                    labels = self.get_labels(elems)
                    if idx % 5000 == 0:
                        print("Encoding sample no: ", idx)

                    encoding_input = (skimage.img_as_uint(x) for x in features_to_bytes)

                    result = sess.run(
                        list(encoding_ops),
                        feed_dict=dict(
                            zip(encoding_placeholders, encoding_input)
                        ))
                    self.save_example_op(writer, *result, *labels)


def get_image_shape(image):
    return image.shape[0], image.shape[1], image.shape[2]


class NotEncodingTFRecordSaver(ABC, AbstractTFRecordSaver):

    def _save_to_tfrecord(self, features, labels, path):
        with tf.python_io.TFRecordWriter(str(path)) as writer:
            for idx, elems in \
                    enumerate(zip(*list(features.values()), *list(labels.values()))):
                features_to_bytes = self.get_features_to_process(elems)
                labels = self.get_labels(elems)
                features_as_bytes = ((skimage.img_as_float(x).tostring()) for x in features_to_bytes)
                self.save_example_op(writer, *features_as_bytes, *labels)

    def resolve_additional_features(self, features, labels):
        self.rows, self.cols, self.depth = get_image_shape(list(features.values())[0][0])

    def add_additional_features(self, features):
        features.update({
            consts.TFRECORD_HEIGHT: _int64_feature(self.rows),
            consts.TFRECORD_WEIGHT: _int64_feature(self.cols),
            consts.TFRECORD_DEPTH: _int64_feature(self.depth)
        })
        return features


class PairedTFRecordSaver(ABC, AbstractTFRecordSaver):
    def validate_data_shape(self, features, labels):
        left_images = np.array(list(features.values())[0])
        right_images = np.array(list(features.values())[1])
        left_labels = labels[consts.LEFT_FEATURE_LABEL]
        right_labels = labels[consts.RIGHT_FEATURE_LABEL]
        same_labels = labels[consts.PAIR_LABEL]

        if left_images.shape[0] != right_images.shape[0] != len(left_labels) != len(right_labels) != len(same_labels):
            raise ValueError('Images size {} and {} does not match labels size {}, {}, {}.'.format(left_images.shape[0],
                                                                                                   right_images.shape[
                                                                                                       0],
                                                                                                   len(left_labels),
                                                                                                   len(right_labels),
                                                                                                   len(same_labels)))
        if left_images.shape != right_images.shape:
            raise ValueError('Left and right images have different shapes!')

        # left = left_images[0]
        # return left.shape[0], left.shape[1], left.shape[2]

    def save_example_op(self, writer: TFRecordWriter, left_bytes: tf.Tensor, right_bytes: tf.Tensor, pair_label: int,
                        left_label: int, right_label: int):
        features = {
            consts.TFRECORD_LEFT_BYTES: _bytes_feature(left_bytes),
            consts.TFRECORD_RIGHT_BYTES: _bytes_feature(right_bytes),
            consts.TFRECORD_PAIR_LABEL: _int64_feature(int(pair_label)),
            consts.TFRECORD_LEFT_LABEL: _int64_feature(int(left_label)),
            consts.TFRECORD_RIGHT_LABEL: _int64_feature(int(right_label)),
        }

        self._save_example(features, writer)

    def get_features_to_process(self, elems):
        (left, right, pair_label, left_label, right_label) = elems
        return left, right

    def get_labels(self, elems):
        (left, right, pair_label, left_label, right_label) = elems
        return pair_label, left_label, right_label


class UnpairedTFRecordSaver(ABC, AbstractTFRecordSaver):
    def validate_data_shape(self, features, labels):
        images = features[consts.FEATURES]
        labels = labels[consts.LABELS]

        if images.shape[0] != len(labels):
            raise ValueError(
                'Images size {} does not match labels size {}.'.format(images.shape[0], len(labels)))

        # image = images[0]
        # return image.shape[0], image.shape[1], image.shape[2]

    def save_example_op(self, writer: TFRecordWriter, image_bytes: tf.Tensor, label: int):
        features = {
            consts.TFRECORD_IMAGE_BYTES: _bytes_feature(image_bytes),
            consts.TFRECORD_LABEL: _int64_feature(label),
        }

        self._save_example(features, writer)

    def get_features_to_process(self, elems):
        (image, label) = elems
        return image,

    def get_labels(self, elems):
        (image, label) = elems
        return label,


class PairedEncodingTFRecordSaver(EncodingTFRecordSaver, PairedTFRecordSaver):

    def get_encoding_ops(self):
        decoded_image1 = tf.placeholder(tf.uint16)
        decoded_image2 = tf.placeholder(tf.uint16)
        encoding_image1 = tf.image.encode_png(decoded_image1)
        encoding_image2 = tf.image.encode_png(decoded_image2)
        return (decoded_image1, decoded_image2), (encoding_image1, encoding_image2)


class UnpairedEncodingTFRecordSaver(EncodingTFRecordSaver, UnpairedTFRecordSaver):

    def get_encoding_ops(self):
        decoded_image = tf.placeholder(tf.uint16)
        encoding_image = tf.image.encode_png(decoded_image)
        return (decoded_image,), (encoding_image,)


class UnpairedNotEncodingTFRecordSaver(NotEncodingTFRecordSaver, UnpairedTFRecordSaver):
    pass


class PairedNotEncodingTFRecordSaver(NotEncodingTFRecordSaver, PairedTFRecordSaver):
    pass
