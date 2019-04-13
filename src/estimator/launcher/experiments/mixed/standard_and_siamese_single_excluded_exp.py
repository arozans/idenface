from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.estimator.model.siamese_conv_model import MnistSiameseModel
from src.utils import consts


class StandardAndSiameseSingleExcludedExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "standard_and_siamese_single_excluded_exp"


class ExcludedAwareMnistCNNModel(MnistCNNModel):

    @property
    def summary(self) -> str:
        return "cnn"

    def __init__(self, excluded) -> None:
        super().__init__()
        self.excluded = excluded

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.EXCLUDED_KEYS: self.excluded})


class ExcludedAwareSiameseModel(MnistSiameseModel):

    @property
    def summary(self) -> str:
        return "siamese"

    def __init__(self, excluded) -> None:
        super().__init__()
        self.excluded = excluded

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.EXCLUDED_KEYS: self.excluded})


launcher = StandardAndSiameseSingleExcludedExperimentLauncher([
    ExcludedAwareMnistCNNModel([1]),
    ExcludedAwareSiameseModel([1]),
    ExcludedAwareMnistCNNModel([2]),
    ExcludedAwareSiameseModel([2]),
    ExcludedAwareMnistCNNModel([3]),
    ExcludedAwareSiameseModel([3]),
    ExcludedAwareMnistCNNModel([4]),
    ExcludedAwareSiameseModel([4]),
    ExcludedAwareMnistCNNModel([5]),
    ExcludedAwareSiameseModel([5]),
    ExcludedAwareMnistCNNModel([6]),
    ExcludedAwareSiameseModel([6]),
    ExcludedAwareMnistCNNModel([7]),
    ExcludedAwareSiameseModel([7]),
    ExcludedAwareSiameseModel([8]),
    ExcludedAwareMnistCNNModel([8]),
])
