from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.contrastive_model import MnistContrastiveModel
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.softmax_model import MnistSoftmaxModel
from src.utils import consts


class SoftmaxAndContrastiveSingleExcludedExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "softmax_and_contrastive_single_excluded_exp"


class ExcludedAwareMnistSoftmaxModel(MnistSoftmaxModel):

    @property
    def summary(self) -> str:
        return "softmax"

    def __init__(self, excluded) -> None:
        super().__init__()
        self.excluded = excluded

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.EXCLUDED_KEYS: self.excluded})


class ExcludedAwareMnistContrastiveModel(MnistContrastiveModel):

    @property
    def summary(self) -> str:
        return "contrastive"

    def __init__(self, excluded) -> None:
        super().__init__()
        self.excluded = excluded

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.EXCLUDED_KEYS: self.excluded})


launcher = SoftmaxAndContrastiveSingleExcludedExperimentLauncher([
    ExcludedAwareMnistSoftmaxModel([1]),
    ExcludedAwareMnistContrastiveModel([1]),
    ExcludedAwareMnistSoftmaxModel([2]),
    ExcludedAwareMnistContrastiveModel([2]),
    ExcludedAwareMnistSoftmaxModel([3]),
    ExcludedAwareMnistContrastiveModel([3]),
    ExcludedAwareMnistSoftmaxModel([4]),
    ExcludedAwareMnistContrastiveModel([4]),
    ExcludedAwareMnistSoftmaxModel([5]),
    ExcludedAwareMnistContrastiveModel([5]),
    ExcludedAwareMnistSoftmaxModel([6]),
    ExcludedAwareMnistContrastiveModel([6]),
    ExcludedAwareMnistSoftmaxModel([7]),
    ExcludedAwareMnistContrastiveModel([7]),
    ExcludedAwareMnistContrastiveModel([8]),
    ExcludedAwareMnistSoftmaxModel([8]),
])
