from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.siamese_conv_model import MnistSiameseModel
from src.utils import consts


class MnistSiameseBatchSizeExperimentLauncher(ExperimentLauncher):
    @property
    def launcher_name(self):
        return "mnist_siamese_batch_size_exp"


class BatchSizeAwareMnistSiameseModel(MnistSiameseModel):

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


launcher = MnistSiameseBatchSizeExperimentLauncher([
    BatchSizeAwareMnistSiameseModel(32),
    BatchSizeAwareMnistSiameseModel(64),
    BatchSizeAwareMnistSiameseModel(128),
    BatchSizeAwareMnistSiameseModel(256),
    BatchSizeAwareMnistSiameseModel(512),
    BatchSizeAwareMnistSiameseModel(700),
    BatchSizeAwareMnistSiameseModel(1024),
    # BatchSizeAwareMnistSiameseModel(300), deduced as best compromise
])
