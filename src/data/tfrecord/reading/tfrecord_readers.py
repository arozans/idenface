from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from src.utils import consts


def normalize_image(image):
    image = np.array(image)
    return image - 0.5


class AbstractReader(ABC):

    def get_decode_op(self):
        def inner(serialized):
            features = self.prepare_features()
            parsed_example = tf.parse_single_example(serialized=serialized, features=features)
            images, labels = self.extract_data(parsed_example)
            images = [self._prepare_image(image, parsed_example) for image in images]
            return self.create_dataset_entry(images, labels)

        return inner

    def _prepare_image(self, image, parsed_example):
        image = self._decode_image(image)
        image = self._process_image(image, parsed_example)
        return normalize_image(image)

    @abstractmethod
    def _decode_image(self, image):
        pass

    def _process_image(self, image, parsed_example):
        return image

    def prepare_features(self):
        features = self.get_features()
        additional_features = self.get_additional_features()
        features.update(additional_features)
        return features

    @abstractmethod
    def get_features(self):
        pass

    def get_additional_features(self):
        return {}

    @abstractmethod
    def extract_data(self, parsed_example):
        pass

    @abstractmethod
    def create_dataset_entry(self, images, labels):
        pass


class PairedReader(AbstractReader, ABC):
    def get_features(self):
        features = \
            {
                consts.TFRECORD_LEFT_BYTES: tf.FixedLenFeature([], tf.string),
                consts.TFRECORD_RIGHT_BYTES: tf.FixedLenFeature([], tf.string),
                consts.TFRECORD_PAIR_LABEL: tf.FixedLenFeature([], tf.int64),
                consts.TFRECORD_LEFT_LABEL: tf.FixedLenFeature([], tf.int64),
                consts.TFRECORD_RIGHT_LABEL: tf.FixedLenFeature([], tf.int64),
            }
        return features

    def extract_data(self, parsed_example):
        left_raw = parsed_example[consts.TFRECORD_LEFT_BYTES]
        right_raw = parsed_example[consts.TFRECORD_RIGHT_BYTES]
        pair_label = parsed_example[consts.TFRECORD_PAIR_LABEL]
        left_label = parsed_example[consts.TFRECORD_LEFT_LABEL]
        right_label = parsed_example[consts.TFRECORD_RIGHT_LABEL]

        images = (left_raw, right_raw)
        labels = (pair_label, left_label, right_label)
        return images, labels

    def create_dataset_entry(self, images, labels):
        return {
                   consts.LEFT_FEATURE_IMAGE: images[0],
                   consts.RIGHT_FEATURE_IMAGE: images[1]
               }, \
               {
                   consts.PAIR_LABEL: labels[0],
                   consts.LEFT_FEATURE_LABEL: labels[1],
                   consts.RIGHT_FEATURE_LABEL: labels[2]
               }


class UnpairedReader(AbstractReader, ABC):

    def get_features(self):
        features = \
            {
                consts.TFRECORD_IMAGE_BYTES: tf.FixedLenFeature([], tf.string),
                consts.TFRECORD_LABEL: tf.FixedLenFeature([], tf.int64),
            }
        return features

    def extract_data(self, parsed_example):
        image_raw = parsed_example[consts.TFRECORD_IMAGE_BYTES]
        label = parsed_example[consts.TFRECORD_LABEL]

        images = (image_raw,)
        labels = (label,)
        return images, labels

    def create_dataset_entry(self, images, labels):
        return {consts.FEATURES: images[0]}, {consts.LABELS: labels[0]}


class DecodingReader(AbstractReader, ABC):
    def _decode_image(self, image):
        return tf.image.decode_image(image, dtype=tf.float32)


class NotDecodingReader(AbstractReader, ABC):
    def _decode_image(self, image):
        image = tf.decode_raw(image, tf.float32)
        return image

    def _process_image(self, image, parsed_example):
        image_shape = tf.stack([parsed_example[consts.TFRECORD_HEIGHT],
                                parsed_example[consts.TFRECORD_WEIGHT],
                                parsed_example[consts.TFRECORD_DEPTH]])
        image = tf.reshape(image, image_shape)
        return image

    def get_additional_features(self):
        return {
            consts.TFRECORD_HEIGHT: tf.FixedLenFeature([], tf.int64),
            consts.TFRECORD_WEIGHT: tf.FixedLenFeature([], tf.int64),
            consts.TFRECORD_DEPTH: tf.FixedLenFeature([], tf.int64)
        }


class PairedDecodingReader(DecodingReader, PairedReader):
    pass


class UnpairedDecodingReader(DecodingReader, UnpairedReader):
    pass


class PairedNotDecodingReader(NotDecodingReader, PairedReader):
    pass


class UnpairedNotDecodingReader(NotDecodingReader, UnpairedReader):
    pass
