from typing import Any, Dict

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.contrastive_model import FmnistContrastiveModel
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.softmax_model import FmnistSoftmaxModel
from src.utils import consts


class SoftmaxAndContrastiveFmnistMultipleExcludedExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "softmax_and_contrastive_fmnist_multiple_excluded_exp"

    @property
    def params(self):
        return {
            consts.GLOBAL_SUFFIX: 'v3',
            consts.TRAIN_STEPS: 7 * 1000
        }


class ExcludedAwareFmnistSoftmaxModel(FmnistSoftmaxModel):

    @property
    def summary(self) -> str:
        return "softmax"

    def __init__(self, excluded) -> None:
        super().__init__()
        self.excluded = excluded

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.EXCLUDED_KEYS: self.excluded})


class ExcludedAwareFmnistContrastiveModel(FmnistContrastiveModel):

    @property
    def summary(self) -> str:
        return "contrastive"

    def __init__(self, excluded) -> None:
        super().__init__()
        self.excluded = excluded

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.EXCLUDED_KEYS: self.excluded})


launcher = SoftmaxAndContrastiveFmnistMultipleExcludedExperimentLauncher([
    ExcludedAwareFmnistSoftmaxModel([2]),
    ExcludedAwareFmnistContrastiveModel([2]),
    ExcludedAwareFmnistSoftmaxModel([1, 2]),
    ExcludedAwareFmnistContrastiveModel([1, 2]),
    ExcludedAwareFmnistSoftmaxModel([5, 6]),
    ExcludedAwareFmnistContrastiveModel([5, 6]),
    ExcludedAwareFmnistSoftmaxModel([1, 2, 3]),
    ExcludedAwareFmnistContrastiveModel([1, 2, 3]),
    ExcludedAwareFmnistSoftmaxModel([1, 2, 3, 4, 5]),
    ExcludedAwareFmnistContrastiveModel([1, 2, 3, 4, 5]),
    ExcludedAwareFmnistSoftmaxModel([1, 2, 3, 4, 5, 6, 7]),
    ExcludedAwareFmnistContrastiveModel([1, 2, 3, 4, 5, 6, 7]),
    ExcludedAwareFmnistSoftmaxModel([8, 9, 0]),
    ExcludedAwareFmnistContrastiveModel([8, 9, 0]),
    ExcludedAwareFmnistSoftmaxModel([5, 6, 7, 8, 9, 0]),
    ExcludedAwareFmnistContrastiveModel([5, 6, 7, 8, 9, 0]),
    ExcludedAwareFmnistSoftmaxModel([1, 2, 5, 6, 7, 8, 9, 0]),
    ExcludedAwareFmnistContrastiveModel([1, 2, 5, 6, 7, 8, 9, 0]),
    ExcludedAwareFmnistSoftmaxModel([1, 2, 3, 4, 5, 6, 9, 0]),
    ExcludedAwareFmnistContrastiveModel([1, 2, 3, 4, 5, 6, 9, 0]),
])
