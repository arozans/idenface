from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.utils import consts


class StandardCnnBatchSizeExperimentLauncher(ExperimentLauncher):
    @property
    def launcher_name(self):
        return "standard_cnn_batch_size_exp"


class BatchSizeAwareMnistCNNModel(MnistCNNModel):

    @property
    def summary(self) -> str:
        return "lr_" + str(self.batch_size)

    def __init__(self, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.BATCH_SIZE: self.batch_size})


launcher = StandardCnnBatchSizeExperimentLauncher([
    BatchSizeAwareMnistCNNModel(5),
    BatchSizeAwareMnistCNNModel(30),
    BatchSizeAwareMnistCNNModel(300),
    BatchSizeAwareMnistCNNModel(512),
])
