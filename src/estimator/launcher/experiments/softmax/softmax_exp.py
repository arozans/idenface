from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.softmax_model import MnistSoftmaxModel, FmnistSoftmaxModel, ExtruderSoftmaxModel


class SoftmaxExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "softmax_exp"


launcher = SoftmaxExperimentLauncher([
    MnistSoftmaxModel(),
    FmnistSoftmaxModel(),
    ExtruderSoftmaxModel()
])
