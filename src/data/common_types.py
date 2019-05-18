from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Tuple, Union, Dict, Any

import numpy as np
from dataclasses import dataclass, replace, field, InitVar

from src.utils import consts, utils


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

    def as_tuple(self):
        return self.width, self.height, self.channels

    @staticmethod
    def from_tuple(dims):
        return ImageDimensions(*dims)

    @staticmethod
    def of(image: Union[np.ndarray, Path, str]):
        if isinstance(image, Path) or isinstance(image, str):
            image = utils.load_image(image)
        return ImageDimensions.from_tuple(np.array(image).shape)


@dataclass(frozen=True)
class DataDescription:
    variant: DatasetVariant
    image_dimensions: ImageDimensions
    classes_count: int
    storage_method: DatasetStorageMethod = DatasetStorageMethod.IN_MEMORY


class AbstractRawDataProvider(ABC):

    @property
    @abstractmethod
    def description(self) -> DataDescription:
        pass

    @abstractmethod
    def get_raw_train(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_raw_test(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_sample_feature(self) -> Union[np.ndarray, Path]:
        return self.get_raw_test()[0][0]


@dataclass(frozen=True)
class DatasetSpec:
    raw_data_provider: AbstractRawDataProvider
    type: DatasetType
    with_excludes: bool
    encoding: bool = True
    paired: bool = True
    repeating_pairs: bool = True
    identical_pairs: bool = False

    def should_resize_raw_data(self):
        demanded_image_dimensions = self.raw_data_provider.description.image_dimensions
        sample_feature = self.raw_data_provider.get_sample_feature()
        actual_image_dimensions = ImageDimensions.of(sample_feature)
        return demanded_image_dimensions != actual_image_dimensions


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
EXTRUDER_REDUCED_SIZE_DATA_DESCRIPTION = replace(EXTRUDER_DATA_DESCRIPTION,
                                                 image_dimensions=replace(EXTRUDER_DATA_DESCRIPTION.image_dimensions,
                                                                          width=consts.EXTRUDER_REDUCED_SIZE_IMAGE_SIDE_PIXEL_COUNT,
                                                                          height=consts.EXTRUDER_REDUCED_SIZE_IMAGE_SIDE_PIXEL_COUNT))


@dataclass
class DatasetFragment:  # fixme: change name - include 'raw'
    features: np.ndarray
    labels: np.ndarray


class LabelsDict(dict):
    @property
    def all(self):
        return self[consts.LABELS]

    @property
    def pair(self):
        return self[consts.PAIR_LABEL]

    @property
    def left(self):
        return self[consts.LEFT_FEATURE_LABEL]

    @property
    def right(self):
        return self[consts.RIGHT_FEATURE_LABEL]


class FeaturesDict(dict):
    @property
    def all(self):
        return self[consts.FEATURES]

    @property
    def left(self):
        return self[consts.LEFT_FEATURE_IMAGE]

    @property
    def right(self):
        return self[consts.RIGHT_FEATURE_IMAGE]


@dataclass
class DictsDataset:
    features: FeaturesDict = field(init=False)
    labels: LabelsDict = field(init=False)
    features_tmp: InitVar[Dict[str, Any]]
    labels_tmp: InitVar[Dict[str, Any]]

    def __post_init__(self, features_tmp, labels_tmp):
        self.features = FeaturesDict(features_tmp)
        self.labels = LabelsDict(labels_tmp)
