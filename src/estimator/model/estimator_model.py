from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import numpy as np
import tensorflow as tf

from src.data.common_types import AbstractRawDataProvider
from src.estimator.training.supplying_datasets import AbstractDatasetProvider, TFRecordDatasetProvider
from src.utils import consts, utils


def merge_two_dicts(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, Any]:
    z = x.copy()
    z.update(y)
    return z


class EstimatorModel(ABC):
    @abstractmethod
    def get_model_fn(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def summary(self) -> str:
        pass

    @property
    def params(self) -> Dict[str, Any]:
        base_params = {
            consts.DATASET_PROVIDER_CLS: self.dataset_provider_cls,
            consts.RAW_DATA_PROVIDER_CLS: self.get_raw_dataset_provider_cls(),
            # consts.FULL_PROVIDER: self.get_dataset_provider, fixme
        }
        return merge_two_dicts(base_params, self.additional_model_params)

    @property
    @abstractmethod
    def additional_model_params(self) -> Dict[str, Any]:
        return {}

    @property
    @abstractmethod
    def raw_data_provider_cls(self) -> Type[AbstractRawDataProvider]:
        pass

    @property
    def dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return TFRecordDatasetProvider

    @abstractmethod
    def get_predicted_labels(self, result: Dict[str, np.ndarray]):
        pass

    @abstractmethod
    def get_predicted_scores(self, result: Dict[str, np.ndarray]):
        pass

    def get_raw_dataset_provider_cls(self):
        return self.raw_data_provider_cls

    def get_dataset_provider(self):
        return self.dataset_provider_cls(self.get_raw_dataset_provider_cls())

    @property
    def produces_2d_embedding(self) -> bool:
        return False


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
