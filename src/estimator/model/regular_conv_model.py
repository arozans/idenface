from typing import Dict, Any, Type

import numpy as np
import tensorflow as tf

from src.data.common_types import AbstractRawDataProvider
from src.data.raw_data.raw_data_providers import MnistRawDataProvider
from src.estimator.model.estimator_model import EstimatorModel, non_streaming_accuracy
from src.utils import utils, consts
from src.utils.configuration import config


class MnistCNNModel(EstimatorModel):

    @property
    def name(self) -> str:
        return "standardCNN"

    @property
    def summary(self) -> str:
        return self.name

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return {
            consts.BATCH_SIZE: 30,
            consts.OPTIMIZER: 'GradientDescent',
            consts.LEARNING_RATE: 0.15,
            consts.TRAIN_STEPS: 5 * 1000,
            consts.EVAL_STEPS_INTERVAL: 700
        }

    @property
    def raw_data_provider_cls(self) -> Type[AbstractRawDataProvider]:
        return MnistRawDataProvider

    def get_model_fn(self):
        return cnn_model_fn

    def get_predicted_labels(self, result: Dict[str, np.ndarray]):
        return result[consts.INFERENCE_CLASSES]

    def get_predicted_scores(self, result: Dict[str, np.ndarray]):
        return np.max(result['probabilities'], axis=1)


def cnn_model_fn(features, labels, mode, params=None):
    utils.log('Creating graph wih mode: {}'.format(mode))

    with tf.name_scope('left_cnn_stack'):
        flatten_left_stack = create_cnn_layers(features[consts.LEFT_FEATURE_IMAGE])
    with tf.name_scope('right_cnn_stack'):
        flatten_right_stack = create_cnn_layers(features[consts.RIGHT_FEATURE_IMAGE])

    flatted_concat = tf.concat(axis=1, values=[flatten_left_stack, flatten_right_stack])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=flatted_concat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 2]
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    pair_labels = labels[consts.PAIR_LABEL]
    loss = tf.losses.sparse_softmax_cross_entropy(labels=pair_labels, logits=logits)

    accuracy_metric = tf.metrics.accuracy(labels=pair_labels, predictions=predictions["classes"],
                                          name='accuracy_metric')
    recall_metric = tf.metrics.recall(labels=pair_labels, predictions=predictions["classes"], name='recall_metric')
    precision_metric = tf.metrics.precision(labels=pair_labels, predictions=predictions["classes"],
                                            name='precision_metric')
    f1_metric = tf.contrib.metrics.f1_score(labels=pair_labels, predictions=predictions["classes"], name='f1metric')
    train_accuracy = non_streaming_accuracy(predictions["classes"], pair_labels)
    eval_metric_ops = {
        "accuracy": accuracy_metric,
        "recall": recall_metric,
        "precision": precision_metric,
        "f1_metric": f1_metric,
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('accuracy', train_accuracy)

        optimizer = determine_optimizer(config.optimizer)(config.learning_rate)  # fix params to get from flags
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_or_create_global_step())

        logging_hook = tf.train.LoggingTensorHook(
            {
                "accuracy_logging": train_accuracy,
            }, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


def create_cnn_layers(image):
    # reshaping
    flat_image = tf.reshape(image, [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=flat_image,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    return pool2_flat


def determine_optimizer(optimizer_param):
    if optimizer_param == 'GradientDescent':
        return tf.train.GradientDescentOptimizer
    elif optimizer_param == 'AdamOptimizer':
        return tf.train.AdamOptimizer
    else:
        raise ValueError("Unknown optimizer: {}".format(optimizer_param))
