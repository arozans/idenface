from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.utils import consts


class StandardCnnSingleExcludedExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "standard_cnn_single_excluded_exp"


class ExcludedAwareMnistCNNModel(MnistCNNModel):

    @property
    def summary(self) -> str:
        return ""

    def __init__(self, excluded) -> None:
        super().__init__()
        self.excluded = excluded

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.EXCLUDED_KEYS: self.excluded})


launcher = StandardCnnSingleExcludedExperimentLauncher([
    ExcludedAwareMnistCNNModel([1]),
    ExcludedAwareMnistCNNModel([2]),
    ExcludedAwareMnistCNNModel([3]),
    ExcludedAwareMnistCNNModel([4]),
    ExcludedAwareMnistCNNModel([5]),
    ExcludedAwareMnistCNNModel([6]),
    ExcludedAwareMnistCNNModel([7]),
    ExcludedAwareMnistCNNModel([8]),
])
