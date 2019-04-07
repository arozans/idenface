from pathlib import Path

import tensorflow as tf
from tensorflow.python.data import TFRecordDataset

from src.utils import utils, consts

encoding = False


def _prepare_image(image, image_shape):
    if not encoding:
        image = tf.decode_raw(image, tf.float32)
    elif encoding:
        image = tf.image.decode_image(image, dtype=tf.uint16)
        image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.reshape(image, image_shape)
    image = image - 0.5  # normalize - image is already float
    return image


def _decode(serialized):
    features = \
        {
            consts.TFRECORD_LEFT_BYTES: tf.FixedLenFeature([], tf.string),
            consts.TFRECORD_RIGHT_BYTES: tf.FixedLenFeature([], tf.string),
            consts.TFRECORD_PAIR_LABEL: tf.FixedLenFeature([], tf.int64),
            consts.TFRECORD_LEFT_LABEL: tf.FixedLenFeature([], tf.int64),
            consts.TFRECORD_RIGHT_LABEL: tf.FixedLenFeature([], tf.int64),
            consts.TFRECORD_HEIGHT: tf.FixedLenFeature([], tf.int64),
            consts.TFRECORD_WEIGHT: tf.FixedLenFeature([], tf.int64),
            consts.TFRECORD_DEPTH: tf.FixedLenFeature([], tf.int64)
        }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    # Get the image as raw bytes.
    left_raw = parsed_example[consts.TFRECORD_LEFT_BYTES]
    right_raw = parsed_example[consts.TFRECORD_RIGHT_BYTES]
    pair_label = parsed_example[consts.TFRECORD_PAIR_LABEL]
    left_label = parsed_example[consts.TFRECORD_LEFT_LABEL]
    right_label = parsed_example[consts.TFRECORD_RIGHT_LABEL]
    image_shape = tf.stack([parsed_example[consts.TFRECORD_HEIGHT],
                            parsed_example[consts.TFRECORD_WEIGHT],
                            parsed_example[consts.TFRECORD_DEPTH]])
    # Decode the raw bytes so it becomes a tensor with type.
    left_image = _prepare_image(left_raw, image_shape)
    right_image = _prepare_image(right_raw, image_shape)

    d = {consts.LEFT_FEATURE_IMAGE: left_image, consts.RIGHT_FEATURE_IMAGE: right_image}, \
        {consts.PAIR_LABEL: pair_label, consts.LEFT_FEATURE_LABEL: left_label, consts.RIGHT_FEATURE_LABEL: right_label}
    return d


def assemble_dataset(input_data_dir: Path, load_encoded: bool) -> TFRecordDataset:
    global encoding
    encoding = load_encoded

    def all_names_in_dir(dir):
        return [str(f) for f in dir.iterdir()][0]  # only one file atm

    files_to_assemble = all_names_in_dir(input_data_dir)

    assert ((consts.NOT_ENCODED_FILENAME_MARKER not in files_to_assemble) == load_encoded)

    utils.log('Assembling dataset from .tfrecord file(s): {}, encoding: {}'.format(files_to_assemble, encoding))
    dataset = tf.data.TFRecordDataset(filenames=files_to_assemble)
    dataset = dataset.map(_decode, num_parallel_calls=64)

    return dataset
