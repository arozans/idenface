from typing import Type

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.softmax_model import MnistSoftmaxModel
from src.estimator.training.supplying_datasets import AbstractDatasetProvider, FromGeneratorDatasetProvider, \
    TFRecordDatasetProvider


class DatasetProviderExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "softmax_dataset_providers_exp"


class DatasetProviderAwareMnistSoftmaxModel(MnistSoftmaxModel):

    @property
    def summary(self) -> str:
        return "provider_" + str(self.dataset_provider_cls)

    def __init__(self) -> None:
        super().__init__()


class TFRecordMnistSoftmaxModel(DatasetProviderAwareMnistSoftmaxModel):

    @property
    def _dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return TFRecordDatasetProvider


class OnlineRecordMnistSoftmaxModel(DatasetProviderAwareMnistSoftmaxModel):

    @property
    def _dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return FromGeneratorDatasetProvider


launcher = DatasetProviderExperimentLauncher([
    TFRecordMnistSoftmaxModel(),
    OnlineRecordMnistSoftmaxModel(),
])
