from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.softmax_model import MnistSoftmaxModel
from src.utils import consts


class SoftmaxSingleExcludedExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "softmax_single_excluded_exp"


class ExcludedAwareMnistSoftmaxModel(MnistSoftmaxModel):

    @property
    def summary(self) -> str:
        return ""

    def __init__(self, excluded) -> None:
        super().__init__()
        self.excluded = excluded

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.EXCLUDED_KEYS: self.excluded})


launcher = SoftmaxSingleExcludedExperimentLauncher([
    ExcludedAwareMnistSoftmaxModel([1]),
    ExcludedAwareMnistSoftmaxModel([2]),
    ExcludedAwareMnistSoftmaxModel([3]),
    ExcludedAwareMnistSoftmaxModel([4]),
    ExcludedAwareMnistSoftmaxModel([5]),
    ExcludedAwareMnistSoftmaxModel([6]),
    ExcludedAwareMnistSoftmaxModel([7]),
    ExcludedAwareMnistSoftmaxModel([8]),
])
