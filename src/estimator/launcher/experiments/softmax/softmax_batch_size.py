from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.softmax_model import MnistSoftmaxModel
from src.utils import consts


class SoftmaxBatchSizeExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "softmax_batch_size_exp"


class BatchSizeAwareMnistSoftmaxModel(MnistSoftmaxModel):

    @property
    def summary(self) -> str:
        return "lr_" + str(self.batch_size)

    def __init__(self, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.BATCH_SIZE: self.batch_size})


launcher = SoftmaxBatchSizeExperimentLauncher([
    BatchSizeAwareMnistSoftmaxModel(5),
    BatchSizeAwareMnistSoftmaxModel(30),
    BatchSizeAwareMnistSoftmaxModel(300),
    BatchSizeAwareMnistSoftmaxModel(512),
])
