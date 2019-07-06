from typing import Type

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.softmax_model import MnistSoftmaxModel
from src.estimator.training.supplying_datasets import AbstractDatasetProvider, TFRecordDatasetProvider


class MnistEncodingExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "softmax_encoding_exp"


class EncodingVersionAwareMnistSoftmaxModel(MnistSoftmaxModel):

    @property
    def summary(self) -> str:
        return "with_encoding_" + str(self.encoding)

    def __init__(self) -> None:
        super().__init__()

    @property
    def encoding(self):
        pass


class TFRecordNoEncodingDatasetProvider(TFRecordDatasetProvider):

    def is_encoded(self):
        return False


class NoEncodingMnistSoftmaxModel(EncodingVersionAwareMnistSoftmaxModel):
    @property
    def encoding(self):
        return False

    @property
    def _dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return TFRecordNoEncodingDatasetProvider


class TFRecordWithEncodingDatasetProvider(TFRecordDatasetProvider):

    def is_encoded(self):
        return True


class EncodingMnistSoftmaxModel(EncodingVersionAwareMnistSoftmaxModel):

    @property
    def encoding(self):
        return True

    @property
    def _dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return TFRecordWithEncodingDatasetProvider


launcher = MnistEncodingExperimentLauncher([
    EncodingMnistSoftmaxModel(),
    NoEncodingMnistSoftmaxModel(),
])
