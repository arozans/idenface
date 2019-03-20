from pathlib import Path

import tensorflow as tf
from tensorflow.python.data import TFRecordDataset

from src.utils import utils, consts


def _prepare_image(image, image_shape):
    image = tf.decode_raw(image, tf.float32)
    image = tf.reshape(image, image_shape)
    image = image - 0.5  # normalize - image is already float

    return image


def _decode(serialized):
    features = \
        {
            'left_raw': tf.FixedLenFeature([], tf.string),
            'right_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    # Get the image as raw bytes.
    left_raw = parsed_example['left_raw']
    right_raw = parsed_example['right_raw']
    label = parsed_example['label']
    image_shape = tf.stack([parsed_example['height'], parsed_example['width'], parsed_example['depth']])
    # Decode the raw bytes so it becomes a tensor with type.
    left_image = _prepare_image(left_raw, image_shape)
    right_image = _prepare_image(right_raw, image_shape)

    d = {consts.LEFT_FEATURE_IMAGE: left_image, consts.RIGHT_FEATURE_IMAGE: right_image}, label
    return d


def assemble_dataset(input_data_dir: Path) -> TFRecordDataset:
    def all_names_in_dir(dir):
        return [str(f) for f in dir.iterdir()][0]  # only one file atm

    utils.log('Assembling dataset from .tfrecord file(s): {}'.format(all_names_in_dir(input_data_dir)))
    dataset = tf.data.TFRecordDataset(filenames=all_names_in_dir(input_data_dir))
    dataset = dataset.map(_decode, num_parallel_calls=64)

    return dataset
