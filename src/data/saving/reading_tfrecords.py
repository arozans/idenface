from pathlib import Path

import tensorflow as tf
from tensorflow.python.data import TFRecordDataset

from src.data.common_types import DatasetSpec
from src.utils import utils, consts


def _prepare_image(image, image_shape, encoding):
    if not encoding:
        image = tf.decode_raw(image, tf.float32)
        image = tf.reshape(image, image_shape)
    elif encoding:
        image = tf.image.decode_image(image, dtype=tf.uint16)
        image = tf.image.convert_image_dtype(image, tf.float32)

    image = image - 0.5  # normalize - image is already float
    return image


def assemble_dataset(input_data_dir: Path, dataset_spec: DatasetSpec) -> TFRecordDataset:
    encoding = dataset_spec.encoding

    def all_names_in_dir(dir):
        return [str(f) for f in dir.iterdir()][0]  # only one file atm

    files_to_assemble = all_names_in_dir(input_data_dir)

    assert ((consts.INPUT_DATA_RAW_DIR_FRAGMENT not in files_to_assemble) == dataset_spec.encoding)

    utils.log('Assembling dataset from .tfrecord file(s): {}, encoding: {}'.format(files_to_assemble, encoding))
    dataset = tf.data.TFRecordDataset(filenames=files_to_assemble)

    decode_op = resolve_decode_op(dataset_spec)
    dataset = dataset.map(decode_op, num_parallel_calls=64)

    return dataset


def get_image_shape_features_names(parsed_example, encoding):
    if not encoding:
        image_shape = tf.stack([parsed_example[consts.TFRECORD_HEIGHT],
                                parsed_example[consts.TFRECORD_WEIGHT],
                                parsed_example[consts.TFRECORD_DEPTH]])
    else:
        image_shape = None
    return image_shape


image_size_features_dict = {
    consts.TFRECORD_HEIGHT: tf.FixedLenFeature([], tf.int64),
    consts.TFRECORD_WEIGHT: tf.FixedLenFeature([], tf.int64),
    consts.TFRECORD_DEPTH: tf.FixedLenFeature([], tf.int64)
}


def resolve_decode_op(dataset_spec):
    encoding = dataset_spec.encoding

    def _decode_pairs(serialized):
        features = \
            {
                consts.TFRECORD_LEFT_BYTES: tf.FixedLenFeature([], tf.string),
                consts.TFRECORD_RIGHT_BYTES: tf.FixedLenFeature([], tf.string),
                consts.TFRECORD_PAIR_LABEL: tf.FixedLenFeature([], tf.int64),
                consts.TFRECORD_LEFT_LABEL: tf.FixedLenFeature([], tf.int64),
                consts.TFRECORD_RIGHT_LABEL: tf.FixedLenFeature([], tf.int64),
            }
        if not encoding:
            features.update(image_size_features_dict)
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        # Get the image as raw bytes.
        left_raw = parsed_example[consts.TFRECORD_LEFT_BYTES]
        right_raw = parsed_example[consts.TFRECORD_RIGHT_BYTES]
        pair_label = parsed_example[consts.TFRECORD_PAIR_LABEL]
        left_label = parsed_example[consts.TFRECORD_LEFT_LABEL]
        right_label = parsed_example[consts.TFRECORD_RIGHT_LABEL]

        image_shape = get_image_shape_features_names(parsed_example, encoding)

        left_image = _prepare_image(left_raw, image_shape, encoding)
        right_image = _prepare_image(right_raw, image_shape, encoding)

        d = {consts.LEFT_FEATURE_IMAGE: left_image, consts.RIGHT_FEATURE_IMAGE: right_image}, \
            {consts.PAIR_LABEL: pair_label, consts.LEFT_FEATURE_LABEL: left_label,
             consts.RIGHT_FEATURE_LABEL: right_label}
        return d

    def _decode_unpaired(serialized):
        features = \
            {
                consts.TFRECORD_IMAGE_BYTES: tf.FixedLenFeature([], tf.string),
                consts.TFRECORD_LABEL: tf.FixedLenFeature([], tf.int64),
            }
        if not encoding:
            features.update(image_size_features_dict)
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)

        image_raw = parsed_example[consts.TFRECORD_IMAGE_BYTES]
        label = parsed_example[consts.TFRECORD_LABEL]
        image_shape = get_image_shape_features_names(parsed_example, encoding)
        # Decode the raw bytes so it becomes a tensor with type.
        image = _prepare_image(image_raw, image_shape, encoding)

        d = {consts.FEATURES: image}, {consts.LABELS: label}
        return d

    return _decode_pairs if dataset_spec.paired else _decode_unpaired
