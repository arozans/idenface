from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Type

import numpy as np
from dataclasses import dataclass

from src.utils import consts


class DatasetType(Enum):
    TRAIN = 'train'
    TEST = 'test'


class DatasetVariant(Enum):
    MNIST = 'mnist'
    FMNIST = 'fmnist'


@dataclass(frozen=True)
class DataDescription:
    variant: DatasetVariant
    image_side_length: int
    classes_count: int


class AbstractRawDataProvider(ABC):

    @staticmethod
    @abstractmethod
    def description() -> DataDescription:
        pass

    @abstractmethod
    def get_raw_train(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_raw_test(self) -> Tuple[np.ndarray, np.ndarray]:
        pass


@dataclass(frozen=True)
class DatasetSpec:
    raw_data_provider_cls: Type[AbstractRawDataProvider]
    type: DatasetType
    with_excludes: bool
    encoding: bool = True
    paired: bool = True
    repeating_pairs: bool = True
    identical_pairs: bool = False


MNIST_DATA_DESCRIPTION = DataDescription(variant=DatasetVariant.MNIST,
                                         image_side_length=consts.MNIST_IMAGE_SIDE_PIXEL_COUNT,
                                         classes_count=consts.MNIST_IMAGE_CLASSES_COUNT)
FMNIST_DATA_DESCRIPTION = DataDescription(variant=DatasetVariant.FMNIST,
                                          image_side_length=consts.MNIST_IMAGE_SIDE_PIXEL_COUNT,
                                          classes_count=consts.MNIST_IMAGE_CLASSES_COUNT)
