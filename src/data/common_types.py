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


@dataclass
class ImageDimensions:
    width: int
    height: int = None
    channels: int = 1

    def __post_init__(self):
        if self.height is None:
            self.height = self.width


@dataclass(frozen=True)
class DataDescription:
    variant: DatasetVariant
    image_dimensions: ImageDimensions
    classes_count: int
    storage_method: DatasetStorageMethod = DatasetStorageMethod.IN_MEMORY


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
                                         classes_count=consts.MNIST_IMAGE_CLASSES_COUNT,
                                         image_dimensions=ImageDimensions(consts.MNIST_IMAGE_SIDE_PIXEL_COUNT)
                                         )
FMNIST_DATA_DESCRIPTION = DataDescription(variant=DatasetVariant.FMNIST,
                                          classes_count=consts.MNIST_IMAGE_CLASSES_COUNT,
                                          image_dimensions=ImageDimensions(consts.MNIST_IMAGE_SIDE_PIXEL_COUNT))
EXTRUDER_DATA_DESCRIPTION = DataDescription(variant=DatasetVariant.EXTRUDER,
                                            classes_count=consts.EXTRUDER_IMAGE_CLASSES_COUNT,
                                            storage_method=DatasetStorageMethod.ON_DISC,
                                            image_dimensions=ImageDimensions(consts.EXTRUDER_IMAGE_SIDE_PIXEL_COUNT,
                                                                             channels=3))


@dataclass
class DatasetFragment:
    features: np.ndarray
    labels: np.ndarray
