from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.contrastive_model import MnistContrastiveModel, FmnistContrastiveModel, \
    ExtruderContrastiveModel


class ContrastiveExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "contrastive_exp"


launcher = ContrastiveExperimentLauncher([
    MnistContrastiveModel(),
    FmnistContrastiveModel(),
    ExtruderContrastiveModel()
])
