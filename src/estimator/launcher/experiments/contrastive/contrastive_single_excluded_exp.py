from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.contrastive_model import MnistContrastiveModel
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.utils import consts


class ContrastiveSingleExcludedExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "contrastive_single_excluded_exp"


class ExcludedAwareMnistContrastiveModel(MnistContrastiveModel):

    @property
    def summary(self) -> str:
        return ""

    def __init__(self, excluded) -> None:
        super().__init__()
        self.excluded = excluded

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.EXCLUDED_KEYS: self.excluded})


launcher = ContrastiveSingleExcludedExperimentLauncher([
    ExcludedAwareMnistContrastiveModel([1]),
    ExcludedAwareMnistContrastiveModel([2]),
    ExcludedAwareMnistContrastiveModel([3]),
    ExcludedAwareMnistContrastiveModel([4]),
    ExcludedAwareMnistContrastiveModel([5]),
    ExcludedAwareMnistContrastiveModel([6]),
    ExcludedAwareMnistContrastiveModel([7]),
    ExcludedAwareMnistContrastiveModel([8]),
])
