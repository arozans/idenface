from typing import Dict, Any, Type

import numpy as np
import tensorflow as tf

from helpers.test_helpers import FakeRawDataProvider
from src.data.common_types import AbstractRawDataProvider
from src.estimator.model.estimator_model import EstimatorModel
from src.utils import utils, consts
from src.utils.configuration import config


class FakeModel(EstimatorModel):
    def get_predicted_labels(self, result: np.ndarray):
        pass

    def get_predicted_scores(self, result: np.ndarray):
        pass

    @property
    def dataset_provider_cls(self) -> Type[AbstractRawDataProvider]:
        return self._data_provider

    def __init__(self, data_provider=FakeRawDataProvider):
        self._data_provider: Type[AbstractRawDataProvider] = data_provider
        self.model_fn_calls = 0

    @property
    def name(self) -> str:
        return "fakeCNN"

    @property
    def summary(self) -> str:
        return self.name

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return {}

    def get_model_fn(self):
        return self.fake_cnn_model_fn

    def fake_cnn_model_fn(self, features, labels, mode, params):
        utils.log('Creating graph wih mode: {}'.format(mode))
        self.model_fn_calls += 1
        with tf.name_scope('left_cnn_stack'):
            flatten_left_stack = self.create_simple_cnn_layers(features[consts.LEFT_FEATURE_IMAGE])
        with tf.name_scope('right_cnn_stack'):
            flatten_right_stack = self.create_simple_cnn_layers(features[consts.RIGHT_FEATURE_IMAGE])

        flatted_concat = tf.concat(axis=1, values=[flatten_left_stack, flatten_right_stack])

        logits = tf.layers.dense(inputs=flatted_concat, units=2)

        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = determine_optimizer(config.optimizer)(config.learning_rate)  # fix params to get from flags
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_or_create_global_step())

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def create_simple_cnn_layers(self, image):
        side_pixel_count = self.dataset_provider_cls.description().image_side_length
        flat_image = tf.reshape(image, [-1, side_pixel_count, side_pixel_count, 1])

        # Convolutional Layer #1
        # Computes 2 features using a 2x2 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, side_pixel_count, side_pixel_count, 1]
        # Output Tensor Shape: [batch_size, side_pixel_count, side_pixel_count, 2]
        conv1 = tf.layers.conv2d(
            inputs=flat_image,
            filters=2,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, side_pixel_count, side_pixel_count, 2]
        # Output Tensor Shape: [batch_size, side_pixel_count/2, side_pixel_count/2, 2]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, TEST_IMAGE_SIDE_PIXEL_COUNT/2, TEST_IMAGE_SIDE_PIXEL_COUNT/2, 2]
        # Output Tensor Shape: [batch_size, TEST_IMAGE_SIDE_PIXEL_COUNT/2 * TEST_IMAGE_SIDE_PIXEL_COUNT/2 * 2]
        pool2_flat = tf.reshape(pool1, [-1, (side_pixel_count // 2 * side_pixel_count // 2 * 2)])
        return pool2_flat


def non_streaming_accuracy(predictions, labels):
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def determine_optimizer(optimizer_param):
    if optimizer_param == 'GradientDescent':
        return tf.train.GradientDescentOptimizer
    elif optimizer_param == 'AdamOptimizer':
        return tf.train.AdamOptimizer
    else:
        raise ValueError("Unknown optimizer: {}".format(optimizer_param))
