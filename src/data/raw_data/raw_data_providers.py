from typing import Tuple

import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.examples.tutorials.mnist import input_data

from src.data.common_types import AbstractRawDataProvider, DataDescription
from src.utils import filenames
from src.utils.consts import MNIST_DATA_DESCRIPTION


class MnistRawDataProvider(AbstractRawDataProvider):

    @staticmethod
    def description() -> DataDescription:
        return MNIST_DATA_DESCRIPTION

    @staticmethod
    def _reshape_into_raw_mnist(images: np.ndarray) -> np.ndarray:
        return images.reshape(-1, 28, 28, 1)

    def get_raw_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """

        Returns:
        train with shape (60.000, 28, 28) and labels (60.000, 1):

        """
        raw = self._get_raw_dataset()
        return self._reshape_into_raw_mnist(raw.train.images), raw.train.labels

    def get_raw_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """

        Returns:
        test with shape (10.000, 28, 28) and labels (10.000, 1):
        """
        raw = self._get_raw_dataset()
        return self._reshape_into_raw_mnist(
            raw.test.images), raw.test.labels

    def _get_raw_dataset(self) -> base.Datasets:
        """
        both train and test are downloaded and separated here.


        Returns:

        base.Datasets containing mnist train, and test datasets. Type is dtypes.float32, so pixels are represented by
        values in range `[0, 1]` .

        train shape is (60.000, 784)

        test shape is (10.000, 784)

        """
        return input_data.read_data_sets(str(filenames.get_raw_input_data_dir()), one_hot=False,
                                         validation_size=0)
