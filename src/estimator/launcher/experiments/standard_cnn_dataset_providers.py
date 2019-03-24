from typing import Type

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.estimator.training.supplying_datasets import AbstractDatasetProvider, FromGeneratorDatasetProvider, \
    TFRecordDatasetProvider


class StandardCnnDatasetProviderExperimentLauncher(ExperimentLauncher):
    @property
    def launcher_name(self):
        return "standard_cnn_dataset_providers_exp"


class DatasetProviderAwareMnistCNNModel(MnistCNNModel):

    @property
    def summary(self) -> str:
        return "provider_" + str(self.dataset_provider_cls)

    def __init__(self) -> None:
        super().__init__()


class TFRecordMnistCNNModel(DatasetProviderAwareMnistCNNModel):

    @property
    def dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return TFRecordDatasetProvider


class OnlineRecordMnistCNNModel(DatasetProviderAwareMnistCNNModel):

    @property
    def dataset_provider_cls(self) -> Type[AbstractDatasetProvider]:
        return FromGeneratorDatasetProvider


launcher = StandardCnnDatasetProviderExperimentLauncher([
    TFRecordMnistCNNModel(),
    OnlineRecordMnistCNNModel(),
])
