from typing import Dict, Any, Type

import numpy as np
import tensorflow as tf

from src.data.common_types import AbstractRawDataProvider
from src.data.raw_data.raw_data_providers import MnistRawDataProvider, FmnistRawDataProvider, ExtruderRawDataProvider
from src.estimator.model import estimator_model
from src.estimator.model.estimator_model import EstimatorModel, non_streaming_accuracy
from src.estimator.training.supplying_datasets import AbstractDatasetProvider, TFRecordDatasetProvider
from src.utils import utils, consts
from src.utils.configuration import config


class MnistCNNModel(EstimatorModel):

    @property
    def name(self) -> str:
        return "standardCNN"

    @property
    def summary(self) -> str:
        return self.name + '_' + str(self.raw_data_provider.description.variant.name)

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return {
            consts.BATCH_SIZE: 300,
            consts.TRAIN_STEPS: 5 * 1000,
            consts.EVAL_STEPS_INTERVAL: 500,
            consts.OPTIMIZER: consts.ADAM_OPTIMIZER,
            consts.LEARNING_RATE: 0.0005
        }

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return MnistRawDataProvider()

    def get_model_fn(self):
        return self.cnn_model_fn

    def get_predicted_labels(self, result: Dict[str, np.ndarray]):
        return result[consts.INFERENCE_CLASSES]

    def get_predicted_scores(self, result: Dict[str, np.ndarray]):
        return np.max(result['probabilities'], axis=1)

    def cnn_model_fn(self, features, labels, mode, params=None):
        utils.log('Creating graph wih mode: {}'.format(mode))

        with tf.name_scope('left_cnn_stack'):
            flatten_left_stack = self.create_cnn_layers(features[consts.LEFT_FEATURE_IMAGE])
        with tf.name_scope('right_cnn_stack'):
            flatten_right_stack = self.create_cnn_layers(features[consts.RIGHT_FEATURE_IMAGE])

        flatted_concat = tf.concat(axis=1, values=[flatten_left_stack, flatten_right_stack])

        dense = tf.layers.dense(inputs=flatted_concat, units=1024, activation=tf.nn.relu)

        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

        logits = tf.layers.dense(inputs=dropout, units=2)

        predictions = {
            consts.INFERENCE_CLASSES: tf.argmax(input=logits, axis=1),
            consts.INFERENCE_SOFTMAX_PROBABILITIES: tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        pair_labels = labels[consts.PAIR_LABEL]
        loss = tf.losses.sparse_softmax_cross_entropy(labels=pair_labels, logits=logits)

        accuracy_metric = tf.metrics.accuracy(labels=pair_labels, predictions=predictions[consts.INFERENCE_CLASSES],
                                              name="accuracy_metric")
        recall_metric = tf.metrics.recall(labels=pair_labels, predictions=predictions[consts.INFERENCE_CLASSES],
                                          name="recall_metric")
        precision_metric = tf.metrics.precision(labels=pair_labels, predictions=predictions[consts.INFERENCE_CLASSES],
                                                name="precision_metric")
        f1_metric = tf.contrib.metrics.f1_score(labels=pair_labels, predictions=predictions[consts.INFERENCE_CLASSES],
                                                name="f1_metric")
        train_accuracy = non_streaming_accuracy(predictions[consts.INFERENCE_CLASSES], pair_labels)
        eval_metric_ops = {
            consts.METRIC_ACCURACY: accuracy_metric,
            consts.METRIC_RECALL: recall_metric,
            consts.METRIC_PRECISION: precision_metric,
            consts.METRIC_F1: f1_metric,
        }

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('accuracy', train_accuracy)

            optimizer = estimator_model.determine_optimizer(config[consts.OPTIMIZER],
                                                            config[consts.LEARNING_RATE])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_or_create_global_step())

            logging_hook = tf.train.LoggingTensorHook(
                {
                    "accuracy_logging": train_accuracy,
                }, every_n_iter=100)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

    def create_cnn_layers(self, image):
        dimensions = self.raw_data_provider.description.image_dimensions

        flat_image = tf.reshape(image, [-1, *dimensions])
        conv1 = tf.layers.conv2d(
            inputs=flat_image,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='same')
        last_filter_size = 64
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=last_filter_size,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='same')
        output = utils.calculate_convmax_output(dimensions.width, 2)
        pool2_flat = tf.reshape(pool2, [-1, output * output * last_filter_size])
        return pool2_flat


class FmnistCNNModel(MnistCNNModel):

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return FmnistRawDataProvider()


class ExtruderCNNModel(MnistCNNModel):

    @property
    def _dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return TFRecordDatasetProvider

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return ExtruderRawDataProvider(100)
