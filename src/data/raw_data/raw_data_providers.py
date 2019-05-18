from pathlib import Path
from typing import Tuple, List

import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.examples.tutorials.mnist import input_data

from src.data.common_types import EXTRUDER_DATA_DESCRIPTION, AbstractRawDataProvider, DataDescription, \
    MNIST_DATA_DESCRIPTION, \
    FMNIST_DATA_DESCRIPTION, DatasetFragment, DatasetType, EXTRUDER_REDUCED_SIZE_DATA_DESCRIPTION
from src.utils import filenames


class MnistRawDataProvider(AbstractRawDataProvider):

    @property
    def description(self) -> DataDescription:
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


class FmnistRawDataProvider(AbstractRawDataProvider):

    @property
    def description(self) -> DataDescription:
        return FMNIST_DATA_DESCRIPTION

    @staticmethod
    def _reshape_into_raw_fmnist(images: np.ndarray) -> np.ndarray:
        return images.reshape(-1, 28, 28, 1)

    def get_raw_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """

        Returns:
        train with shape (60.000, 28, 28) and labels (60.000, 1):

        """
        from tensorflow import keras
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (_, _) = fashion_mnist.load_data()
        return self._reshape_into_raw_fmnist(
            train_images), train_labels

    def get_raw_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """

        Returns:
        test with shape (10.000, 28, 28) and labels (10.000, 1):
        """
        from tensorflow import keras
        fashion_mnist = keras.datasets.fashion_mnist

        (_, _), (test_images, test_labels) = fashion_mnist.load_data()
        return self._reshape_into_raw_fmnist(
            test_images), test_labels


def get_filenames_and_labels(labeled_dirs: List[Path]):
    filenames = []
    labels = []
    for dir in labeled_dirs:
        for elem in dir.iterdir():
            elem = elem.resolve()
            filenames.append(elem)
            labels.append(int(elem.parent.parts[-1]))
    return DatasetFragment(np.array(filenames), np.array(labels))


class ExtruderRawDataProvider(AbstractRawDataProvider):
    def __init__(self):
        self.test_percent = 10

    @property
    def description(self) -> DataDescription:
        return EXTRUDER_DATA_DESCRIPTION

    def get_raw_train(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_dataset_fragment(DatasetType.TRAIN)

    def get_raw_test(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_dataset_fragment(DatasetType.TEST)

    def get_dataset_fragment(self, type: DatasetType) -> Tuple[np.ndarray, np.ndarray]:
        directory: Path = filenames.get_raw_input_data_dir() / self.description.variant.name.lower()
        labeled_dirs = sorted(list(directory.iterdir()))
        assert directory.exists() and len(labeled_dirs) > 0, "Raw data for {} not exists under {}".format(
            self.description.variant.name, directory)
        test_dirs_count = len(labeled_dirs) // 10
        if type == DatasetType.TRAIN:
            labeled_dirs = labeled_dirs[:-test_dirs_count]
        elif type == DatasetType.TEST:
            labeled_dirs = labeled_dirs[-test_dirs_count:]

        dataset_fragment: DatasetFragment = get_filenames_and_labels(labeled_dirs)
        return dataset_fragment.features, dataset_fragment.labels


class ExtruderRawDataReducedSize(ExtruderRawDataProvider):
    @property
    def description(self) -> DataDescription:
        return EXTRUDER_REDUCED_SIZE_DATA_DESCRIPTION
