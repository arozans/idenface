from abc import ABC
from typing import Dict, Any

import numpy as np
import tensorflow as tf

from src.data.common_types import AbstractRawDataProvider
from src.data.raw_data.raw_data_providers import MnistRawDataProvider, FmnistRawDataProvider, ExtruderRawDataProvider
from src.estimator.model import estimator_conv_model
from src.estimator.model.estimator_conv_model import EstimatorConvModel, non_streaming_accuracy, merge_two_dicts
from src.utils import utils, consts, model_params_calc
from src.utils.configuration import config


class SoftmaxModel(EstimatorConvModel, ABC):

    @property
    def name(self) -> str:
        return "softmax"

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(
            super().additional_model_params, {
                consts.BATCH_SIZE: 300,
                consts.TRAIN_STEPS: 5 * 1000,
                consts.EVAL_STEPS_INTERVAL: 500,
                consts.LEARNING_RATE: 0.0005,
                consts.FILTERS: [32, 8],
                consts.KERNEL_SIDE_LENGTHS: [5, 5],
                consts.DENSE_UNITS: [],
                consts.CONCAT_DENSE_UNITS: [4, 2],
                consts.CONCAT_DROPOUT_RATES: [0.5, None],
            })

    def get_model_fn(self):
        return self.softmax_model_fn

    def get_predicted_labels(self, result: Dict[str, np.ndarray]):
        return result[consts.INFERENCE_CLASSES]

    def get_predicted_scores(self, result: Dict[str, np.ndarray]):
        return np.max(result[consts.INFERENCE_SOFTMAX_PROBABILITIES], axis=1)

    def get_parameters_count_dict(self) -> Dict[str, int]:
        dimensions = self.raw_data_provider.description.image_dimensions
        filters = config[consts.FILTERS]
        concat_dense_units = config[consts.CONCAT_DENSE_UNITS]

        conv_output_size = model_params_calc.calculate_convmax_output(dimensions.width,
                                                                      len(config[consts.FILTERS]),
                                                                      config[consts.POOLING_STRIDE])
        param_count = super().get_parameters_count_dict()
        conv_params = param_count[consts.CONV_PARAMS_COUNT]
        dense_params = param_count[consts.DENSE_PARAMS_COUNT]

        concat_dense_params = model_params_calc.calculate_concat_dense_params(conv_output_size, filters,
                                                                              concat_dense_units, 2)
        return {
            consts.CONV_PARAMS_COUNT: conv_params,
            consts.DENSE_PARAMS_COUNT: dense_params,
            consts.CONCAT_DENSE_PARAMS_COUNT: concat_dense_params,
            consts.ALL_PARAMS_COUNT: 2 * (conv_params + dense_params) + concat_dense_params
        }

    def softmax_model_fn(self, features, labels, mode, params=None):
        utils.log('Creating graph wih mode: {}'.format(mode))

        with tf.variable_scope("left_cnn_stack"):
            flatten_left_stack = self.conv_net(features[consts.LEFT_FEATURE_IMAGE], reuse=False)
        with tf.variable_scope("right_cnn_stack"):
            flatten_right_stack = self.conv_net(features[consts.RIGHT_FEATURE_IMAGE], reuse=False)

        with tf.variable_scope("dense_stack"):
            net = tf.concat(axis=1, values=[flatten_left_stack, flatten_right_stack])
            concat_dense_units = config[consts.CONCAT_DENSE_UNITS]
            concat_dropout_rates = config[consts.CONCAT_DROPOUT_RATES]
            assert len(concat_dense_units) == len(concat_dropout_rates)

            for i, (units, rate) in enumerate(zip(concat_dense_units, concat_dropout_rates)):
                net = tf.contrib.layers.fully_connected(
                    inputs=net,
                    num_outputs=units,
                    activation_fn=self.get_activation_fn(i, len(concat_dense_units)))
                if rate:
                    net = tf.layers.dropout(
                        inputs=net,
                        rate=rate,
                        training=(mode == tf.estimator.ModeKeys.TRAIN)
                    )

        predictions = {
            consts.INFERENCE_CLASSES: tf.argmax(input=net, axis=1),
            consts.INFERENCE_SOFTMAX_PROBABILITIES: tf.nn.softmax(net, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        pair_labels = labels[consts.PAIR_LABEL]
        loss = tf.losses.sparse_softmax_cross_entropy(labels=pair_labels, logits=net)

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
            tf.summary.scalar(consts.METRIC_ACCURACY, train_accuracy)

            optimizer = estimator_conv_model.determine_optimizer(config[consts.OPTIMIZER],
                                                                 config[consts.LEARNING_RATE])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_or_create_global_step())

            logging_hook = tf.train.LoggingTensorHook(
                {
                    "accuracy_logging": train_accuracy,
                }, every_n_iter=config[consts.TRAIN_LOG_STEPS_INTERVAL])

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


class MnistSoftmaxModel(SoftmaxModel):

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return MnistRawDataProvider()


class FmnistSoftmaxModel(SoftmaxModel):

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return FmnistRawDataProvider()


class ExtruderSoftmaxModel(SoftmaxModel):

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return ExtruderRawDataProvider(100)
