import io
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Dict

import PIL
import numpy as np
import skimage
import tensorflow as tf
from PIL import Image
from tensorflow.python.lib.io.tf_record import TFRecordWriter

from src.data.common_types import DatasetSpec
from src.utils import consts, utils


def _bytes_feature(value):
    return tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=[value]))


class AbstractSaver:
    def __init__(self, dataset_spec: DatasetSpec):
        self.dataset_spec = dataset_spec

    def save(self, features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], path: Path):
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
        example = tf.compat.v1.train.Example(features=tf.compat.v1.train.Features(feature=features))
        writer.write(example.SerializeToString())

    def preprocess_features(self, features):
        return features


class FromMemoryEncodingSaver(ABC, AbstractSaver):

    @abstractmethod
    def get_encoding_ops(self):
        pass

    def _save_to_tfrecord(self, features, labels, path):
        encoding_placeholders, encoding_ops = self.get_encoding_ops()
        with tf.compat.v1.Session() as sess:
            with TFRecordWriter(str(path)) as writer:
                for idx, elems in \
                        enumerate(zip(*list(features.values()), *list(labels.values()))):
                    features_to_bytes = self.get_features_to_process(elems)
                    labels = self.get_labels(elems)
                    if idx % 5000 == 0:
                        print("encoding sample no: ", idx)

                    encoding_input = (skimage.img_as_uint(x) for x in features_to_bytes)

                    result = sess.run(
                        list(encoding_ops),
                        feed_dict=dict(
                            zip(encoding_placeholders, encoding_input)
                        ))
                    self.save_example_op(writer, *result, *labels)


class RawBytesSaver(ABC, AbstractSaver):

    def _save_to_tfrecord(self, features, labels, path):
        with tf.python_io.TFRecordWriter(str(path)) as writer:
            for idx, elems in \
                    enumerate(zip(*list(features.values()), *list(labels.values()))):
                if idx % 500 == 0:
                    print("Preprocessing value... : ", idx)
                features = self.get_features_to_process(elems)
                labels = self.get_labels(elems)
                features_as_bytes = self.preprocess_features(features)
                self.save_example_op(writer, *features_as_bytes, *labels)


def _to_bytes(param, format):
    ram = io.BytesIO()
    param.save(ram, format=format)
    return ram.getvalue()


def resize_and_save_to_bytes(path_feature, expected_dims):
    image = utils.load_image(path_feature)
    format = image.format
    image = image.resize((expected_dims.height, expected_dims.width), PIL.Image.ANTIALIAS)
    return _to_bytes(image, format)


class FromDiscRawBytesSaver(RawBytesSaver, ABC):
    def __init__(self, dataset_spec: DatasetSpec):
        super().__init__(dataset_spec)
        self.resizing = dataset_spec.should_resize_raw_data()

    def preprocess_features(self, features):
        expected_dims = self.dataset_spec.raw_data_provider.description.image_dimensions
        if self.resizing:
            images = [resize_and_save_to_bytes(x, expected_dims) for x in features]
        else:
            images = [open(x, 'rb').read() for x in features]
        return np.array(images)


def get_image_shape(image):
    return image.shape[0], image.shape[1], image.shape[2]


class FromMemoryRawBytesSaver(RawBytesSaver, ABC):

    def preprocess_features(self, features):
        return ((skimage.img_as_float(x).tostring()) for x in features)

    def resolve_additional_features(self, features, labels):
        self.rows, self.cols, self.depth = get_image_shape(list(features.values())[0][0])

    def add_additional_features(self, features):
        features.update({
            consts.TFRECORD_HEIGHT: _int64_feature(self.rows),
            consts.TFRECORD_WEIGHT: _int64_feature(self.cols),
            consts.TFRECORD_DEPTH: _int64_feature(self.depth)
        })
        return features


class PairedSaver(ABC, AbstractSaver):
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
            raise ValueError('Left and right features have different shapes!')

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
        return (left, right)

    def get_labels(self, elems):
        (left, right, pair_label, left_label, right_label) = elems
        return pair_label, left_label, right_label


class UnpairedSaver(ABC, AbstractSaver):
    def validate_data_shape(self, features, labels):
        images = features[consts.FEATURES]
        labels = labels[consts.LABELS]

        if images.shape[0] != len(labels):
            raise ValueError(
                'Images size {} does not match labels size {}.'.format(images.shape[0], len(labels)))

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


class PairedFromMemoryEncodingSaver(FromMemoryEncodingSaver, PairedSaver):
    # todo: can be used generically, with len(features.values())
    def get_encoding_ops(self):
        decoded_image1 = tf.compat.v1.placeholder(tf.uint16)
        decoded_image2 = tf.compat.v1.placeholder(tf.uint16)
        encoding_image1 = tf.image.encode_png(decoded_image1)
        encoding_image2 = tf.image.encode_png(decoded_image2)
        return (decoded_image1, decoded_image2), (encoding_image1, encoding_image2)


class UnpairedFromMemoryEncodingSaver(FromMemoryEncodingSaver, UnpairedSaver):

    def get_encoding_ops(self):
        decoded_image = tf.compat.v1.placeholder(tf.uint16)
        encoding_image = tf.image.encode_png(decoded_image)
        return (decoded_image,), (encoding_image,)


class UnpairedFromMemoryRawBytesSaver(FromMemoryRawBytesSaver, UnpairedSaver):
    pass


class PairedFromMemoryRawBytesSaver(FromMemoryRawBytesSaver, PairedSaver):
    pass


class UnpairedFromDiscRawBytesSaver(FromDiscRawBytesSaver, UnpairedSaver):
    pass


class PairedFromDiscRawBytesSaver(FromDiscRawBytesSaver, PairedSaver):
    pass
