from enum import Enum, auto
from typing import Type, Tuple

import numpy as np
import tensorflow as tf
from dataclasses import dataclass

from helpers import test_consts
from src.data.common_types import AbstractRawDataProvider, DataDescription
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.estimator.training import training
from src.utils import configuration

STUB_ESTIMATOR_SUMMARY = "stub_estimator_summary"
STUB_ESTIMATOR_NAME = "stub_estimator_name"


class NumberTranslation:
    ONE_TRANS = ("jeden", "uno", "ein")
    TWO_TRANS = ("dwa", "dos", "zwei")
    THREE_TRANS = ("trzy", "tres", "drei")

    def __init__(self, number: int, trans: str):
        self.number = number
        self.trans = trans

    def __repr__(self):
        return "{}({})".format(self.trans, self.number)

    def __hash__(self):
        return hash(self.trans)

    def __eq__(self, other):
        return self.trans == other.trans

    def __ne__(self, other):
        return not self == other


def run_app():
    configuration.define_cli_args()
    try:
        tf.app.run(training.main)
    except(SystemExit):
        print("Test main finished")


class FakeMnistCNNModel(MnistCNNModel):
    @property
    def dataset_provider_cls(self) -> Type[AbstractRawDataProvider]:
        return FakeMnistRawDataProvider


class FakeRawDataProvider(AbstractRawDataProvider):

    def __init__(self):
        self.raw_fake_dataset = RawDataset(
            train=DatasetFragment(
                images=generate_fake_images(size=(test_consts.FAKE_IMAGES_IN_DATASET_COUNT,
                                                  self.description().image_side_length,
                                                  self.description().image_side_length, 1)),
                labels=generate_fake_labels(size=test_consts.FAKE_IMAGES_IN_DATASET_COUNT,
                                            classes=self.description().classes_count)),
            test=DatasetFragment(
                images=generate_fake_images(size=(test_consts.FAKE_IMAGES_IN_DATASET_COUNT,
                                                  self.description().image_side_length,
                                                  self.description().image_side_length, 1)),
                labels=generate_fake_labels(size=test_consts.FAKE_IMAGES_IN_DATASET_COUNT,
                                            classes=self.description().classes_count))
        )

    @staticmethod
    def description() -> DataDescription:
        return DataDescription(TestDatasetVariant.FOO, test_consts.FAKE_IMAGE_SIDE_PIXEL_COUNT,
                               test_consts.FAKE_IMAGES_CLASSES_COUNT)

    def get_raw_train(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.raw_fake_dataset.train.images, self.raw_fake_dataset.train.labels

    def get_raw_test(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.raws_fake_dataset.test.images, self.raw_fake_dataset.test.labels


class FakeMnistRawDataProvider(FakeRawDataProvider):

    @staticmethod
    def description() -> DataDescription:
        return DataDescription(TestDatasetVariant.FAKEMNIST, test_consts.MNIST_IMAGE_SIDE_PIXEL_COUNT,
                               test_consts.MNIST_IMAGES_CLASSES_COUNT)


@dataclass
class DatasetFragment:
    images: np.ndarray
    labels: np.ndarray


@dataclass
class RawDataset:
    train: DatasetFragment
    test: DatasetFragment


class TestDatasetVariant(Enum):
    NUMBERTRANSLATION = auto()
    FOO = auto()
    FAKEMNIST = auto()


def generate_fake_images(size: Tuple[int, ...]):
    return np.random.uniform(size=size).astype(np.float32)


def generate_fake_labels(size: int, classes=10):
    return np.random.randint(low=0, high=classes, size=size).astype(np.int64)
