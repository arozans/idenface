from abc import ABC, abstractmethod
from enum import Enum, auto
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
    EXTRUDER = 'extruder'


class DatasetStorageMethod(Enum):
    IN_MEMORY = auto()
    ON_DISC = auto()


@dataclass(frozen=True)
class DataDescription:
    variant: DatasetVariant
    image_side_length: int
    classes_count: int
    storage_method: DatasetStorageMethod = DatasetStorageMethod.IN_MEMORY
    image_channels: int = 1


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
EXTRUDER_DATA_DESCRIPTION = DataDescription(variant=DatasetVariant.EXTRUDER,
                                            image_side_length=consts.EXTRUDER_IMAGE_SIDE_PIXEL_COUNT,
                                            classes_count=consts.EXTRUDER_IMAGE_CLASSES_COUNT,
                                            image_channels=3,
                                            storage_method=DatasetStorageMethod.ON_DISC)


@dataclass
class DatasetFragment:
    features: np.ndarray
    labels: np.ndarray
