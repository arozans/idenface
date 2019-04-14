from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.regular_conv_model import FmnistCNNModel
from src.estimator.model.siamese_conv_model import FmnistSiameseModel
from src.utils import consts


class StandardAndSiameseFmnistExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "standard_and_siamese_fmnist_exp"

    @property
    def params(self):
        return {
            consts.TRAIN_STEPS: 5,
        }


# and excluded: 1,2

launcher = StandardAndSiameseFmnistExperimentLauncher([
    FmnistCNNModel(),
    FmnistSiameseModel()
])
