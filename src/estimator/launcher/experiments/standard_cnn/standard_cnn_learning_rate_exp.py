from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.regular_conv_model import MnistCNNModel
from src.utils import consts


class StandardCnnLearningRateExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "standard_cnn_learing_rate_exp"


class LearningRateAwareMnistCNNModel(MnistCNNModel):

    @property
    def summary(self) -> str:
        return "lr_" + str(self.learning_rate)

    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.LEARNING_RATE: self.learning_rate})


launcher = StandardCnnLearningRateExperimentLauncher([
    LearningRateAwareMnistCNNModel(0.05),
    LearningRateAwareMnistCNNModel(0.1),
    LearningRateAwareMnistCNNModel(0.15),
    LearningRateAwareMnistCNNModel(0.2),
    LearningRateAwareMnistCNNModel(0.3),
])
