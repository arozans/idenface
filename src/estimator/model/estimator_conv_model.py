from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import numpy as np
import tensorflow as tf

from src.data.common_types import AbstractRawDataProvider
from src.estimator.training.supplying_datasets import AbstractDatasetProvider, TFRecordDatasetProvider
from src.utils import consts, utils
from src.utils.configuration import config


def merge_two_dicts(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, Any]:
    z = x.copy()
    z.update(y)
    return z


class EstimatorConvModel(ABC):
    @abstractmethod
    def get_model_fn(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def summary(self) -> str:
        return self.name + '_' + str(self.raw_data_provider.description.variant.name)

    @property
    def params(self) -> Dict[str, Any]:
        base_params = {
            consts.MODEL_SUMMARY: self.summary,
            consts.DATASET_VARIANT: self.raw_data_provider.description.variant.name,
            consts.DATASET_PROVIDER: self._dataset_provider_cls.__name__
        }
        return merge_two_dicts(base_params, self.additional_model_params)

    @property
    @abstractmethod
    def additional_model_params(self) -> Dict[str, Any]:
        return {}

    @property
    @abstractmethod
    def raw_data_provider(self) -> AbstractRawDataProvider:
        pass

    @property
    def _dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return TFRecordDatasetProvider

    @property
    def dataset_provider(self) -> AbstractDatasetProvider:
        return self._dataset_provider_cls(self.raw_data_provider)

    @abstractmethod
    def get_predicted_labels(self, result: Dict[str, np.ndarray]):
        pass

    @abstractmethod
    def get_predicted_scores(self, result: Dict[str, np.ndarray]):
        pass

    @property
    def produces_2d_embedding(self) -> bool:
        return False

    def _summary_from_dict(self, summary_dict: Dict[str, Any]):
        def remove_whitespaces(elem):
            if isinstance(elem, list):
                return utils.pretty_print_list(elem)
            else:
                return str(elem)

        summary = self.name
        for k, v in summary_dict.items():
            summary = summary + '_' + str(k) + '_' + remove_whitespaces(v)
        return summary

    def conv_net(self, conv_input, reuse=False):
        dimensions = self.raw_data_provider.description.image_dimensions

        net = tf.reshape(conv_input, [-1, *dimensions])
        filters = config[consts.FILTERS]
        kernel_side_lengths = config[consts.KERNEL_SIDE_LENGTHS]
        dense_units = config[consts.DENSE_UNITS] if config[consts.DENSE_UNITS] is not None else []

        assert len(filters) == len(kernel_side_lengths), "Filters and kernels must have same length!"
        layers_num = len(filters)
        with tf.name_scope("cnn_stack"):
            for i, (filter_depth, kernel_size) in enumerate(zip(filters, kernel_side_lengths)):
                with tf.variable_scope("conv" + str(i + 1)) as scope:
                    net = tf.contrib.layers.conv2d(
                        inputs=net,
                        num_outputs=filter_depth,
                        kernel_size=kernel_size,
                        activation_fn=tf.nn.relu if dense_units else self.get_activation_fn(i, layers_num),
                        padding='SAME',
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        scope=scope,
                        reuse=reuse
                    )

                    net = tf.contrib.layers.max_pool2d(
                        inputs=net,
                        kernel_size=config[consts.POOLING_KERNEL_SIDE_LENGTH],
                        stride=config[consts.POOLING_STRIDE],
                        padding='SAME'
                    )

            net = tf.contrib.layers.flatten(net)

            for i, units in enumerate(dense_units):
                net = tf.contrib.layers.fully_connected(
                    inputs=net,
                    num_outputs=units,
                    activation_fn=self.get_activation_fn(i, len(dense_units)))

        return net

    @staticmethod
    def get_activation_fn(i, layers_num):
        if i + 1 < layers_num:
            return tf.nn.relu
        else:
            return None


def non_streaming_accuracy(predictions, labels):
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def determine_optimizer(optimizer_param: str, learning_rate: float):
    utils.log("Creating optimizer: {}, with learning rate: {}".format(optimizer_param, learning_rate))
    if optimizer_param == consts.GRADIENT_DESCEND_OPTIMIZER:
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer_param == consts.MOMENTUM_OPTIMIZER:
        return tf.train.MomentumOptimizer(learning_rate, 0.99, use_nesterov=False)
    elif optimizer_param == consts.NESTEROV_OPTIMIZER:
        return tf.train.MomentumOptimizer(learning_rate, 0.99, use_nesterov=True)
    elif optimizer_param == consts.ADAM_OPTIMIZER:
        return tf.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError("Unknown optimizer: {}".format(optimizer_param))
