from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.siamese_conv_model import MnistSiameseModel
from src.utils import consts


class SiameseSingleExcludedExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "siamese_single_excluded_exp"


class ExcludedAwareSiameseModel(MnistSiameseModel):

    @property
    def summary(self) -> str:
        return ""

    def __init__(self, excluded) -> None:
        super().__init__()
        self.excluded = excluded

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.EXCLUDED_KEYS: self.excluded})


launcher = SiameseSingleExcludedExperimentLauncher([
    ExcludedAwareSiameseModel([1]),
    ExcludedAwareSiameseModel([2]),
    ExcludedAwareSiameseModel([3]),
    ExcludedAwareSiameseModel([4]),
    ExcludedAwareSiameseModel([5]),
    ExcludedAwareSiameseModel([6]),
    ExcludedAwareSiameseModel([7]),
    ExcludedAwareSiameseModel([8]),
])
