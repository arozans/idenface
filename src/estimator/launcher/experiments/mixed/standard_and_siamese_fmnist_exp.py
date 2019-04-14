from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.regular_conv_model import FmnistCNNModel
from src.estimator.model.siamese_conv_model import FmnistSiameseModel


class StandardAndSiameseFmnistExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "standard_and_siamese_fmnist_exp"


# and excluded: 1,2

launcher = StandardAndSiameseFmnistExperimentLauncher([
    FmnistCNNModel(),
    FmnistSiameseModel()
])
