from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.contrastive_model import MnistContrastiveModel
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.utils import consts


class MnistContrastiveBatchSizeExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "mnist_contrastive_batch_size_exp"


class BatchSizeAwareMnistContrastiveModel(MnistContrastiveModel):

    @property
    def summary(self) -> str:
        return "batch_size_" + str(self.batch_size)

    def __init__(self, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params,
                               {
                                   consts.BATCH_SIZE: self.batch_size
                               })


launcher = MnistContrastiveBatchSizeExperimentLauncher([
    BatchSizeAwareMnistContrastiveModel(32),
    BatchSizeAwareMnistContrastiveModel(64),
    BatchSizeAwareMnistContrastiveModel(128),
    BatchSizeAwareMnistContrastiveModel(256),
    BatchSizeAwareMnistContrastiveModel(512),
    BatchSizeAwareMnistContrastiveModel(700),
    BatchSizeAwareMnistContrastiveModel(1024),
    # BatchSizeAwareMnistContrastiveModel(300), deduced as best compromise
])
