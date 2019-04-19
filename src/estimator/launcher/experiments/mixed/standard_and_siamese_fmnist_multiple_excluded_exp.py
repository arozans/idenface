from typing import Any, Dict

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.regular_conv_model import FmnistCNNModel
from src.estimator.model.siamese_conv_model import FmnistSiameseModel
from src.utils import consts


class StandardAndFmnistSiameseFmnistMultipleExcludedExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "standard_and_siamese_fmnist_multiple_excluded_exp"

    @property
    def params(self):
        return {
            consts.GLOBAL_SUFFIX: 'v2',
        }


class ExcludedAwareFmnistCNNModel(FmnistCNNModel):

    @property
    def summary(self) -> str:
        return "cnn"

    def __init__(self, excluded) -> None:
        super().__init__()
        self.excluded = excluded

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.EXCLUDED_KEYS: self.excluded})


class ExcludedAwareFmnistSiameseModel(FmnistSiameseModel):

    @property
    def summary(self) -> str:
        return "siamese"

    def __init__(self, excluded) -> None:
        super().__init__()
        self.excluded = excluded

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.EXCLUDED_KEYS: self.excluded})


launcher = StandardAndFmnistSiameseFmnistMultipleExcludedExperimentLauncher([
    ExcludedAwareFmnistCNNModel([2]),
    ExcludedAwareFmnistSiameseModel([2]),
    ExcludedAwareFmnistCNNModel([1, 2]),
    ExcludedAwareFmnistSiameseModel([1, 2]),
    ExcludedAwareFmnistCNNModel([5, 6]),
    ExcludedAwareFmnistSiameseModel([5, 6]),
    ExcludedAwareFmnistCNNModel([1, 2, 3]),
    ExcludedAwareFmnistSiameseModel([1, 2, 3]),
    ExcludedAwareFmnistCNNModel([1, 2, 3, 4, 5]),
    ExcludedAwareFmnistSiameseModel([1, 2, 3, 4, 5]),
    ExcludedAwareFmnistCNNModel([1, 2, 3, 4, 5, 6, 7]),
    ExcludedAwareFmnistSiameseModel([1, 2, 3, 4, 5, 6, 7]),
    ExcludedAwareFmnistCNNModel([8, 9, 0]),
    ExcludedAwareFmnistSiameseModel([8, 9, 0]),
    ExcludedAwareFmnistCNNModel([5, 6, 7, 8, 9, 0]),
    ExcludedAwareFmnistSiameseModel([5, 6, 7, 8, 9, 0]),
    ExcludedAwareFmnistCNNModel([1, 2, 5, 6, 7, 8, 9, 0]),
    ExcludedAwareFmnistSiameseModel([1, 2, 5, 6, 7, 8, 9, 0]),
    ExcludedAwareFmnistCNNModel([1, 2, 3, 4, 5, 6, 9, 0]),
    ExcludedAwareFmnistSiameseModel([1, 2, 3, 4, 5, 6, 9, 0]),
])
