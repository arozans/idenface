from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.softmax_model import MnistSoftmaxModel
from src.utils import consts


class SoftmaxLearningRateExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "softmax_learing_rate_exp"


class LearningRateAwareMnistSoftmaxModel(MnistSoftmaxModel):

    @property
    def summary(self) -> str:
        return "lr_" + str(self.learning_rate)

    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {consts.LEARNING_RATE: self.learning_rate})


launcher = SoftmaxLearningRateExperimentLauncher([
    LearningRateAwareMnistSoftmaxModel(0.05),
    LearningRateAwareMnistSoftmaxModel(0.1),
    LearningRateAwareMnistSoftmaxModel(0.15),
    LearningRateAwareMnistSoftmaxModel(0.2),
    LearningRateAwareMnistSoftmaxModel(0.3),
])
