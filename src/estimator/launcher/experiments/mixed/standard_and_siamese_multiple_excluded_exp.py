from src.estimator.launcher.experiments.mixed.standard_and_siamese_single_excluded_exp import \
    ExcludedAwareMnistCNNModel, ExcludedAwareSiameseModel
from src.estimator.launcher.launchers import ExperimentLauncher


class StandardAndSiameseMultipleExcludedExperimentLauncher(ExperimentLauncher):
    @property
    def launcher_name(self):
        return "standard_and_siamese_Multiple_excluded_exp"


launcher = StandardAndSiameseMultipleExcludedExperimentLauncher([
    ExcludedAwareMnistCNNModel([1, 2]),
    ExcludedAwareSiameseModel([1, 2]),
    ExcludedAwareMnistCNNModel([5, 6]),
    ExcludedAwareSiameseModel([5, 6]),
    ExcludedAwareMnistCNNModel([1, 2, 3]),
    ExcludedAwareSiameseModel([1, 2, 3]),
    ExcludedAwareMnistCNNModel([1, 2, 3, 4, 5]),
    ExcludedAwareSiameseModel([1, 2, 3, 4, 5]),
    ExcludedAwareMnistCNNModel([1, 2, 3, 4, 5, 6, 7]),
    ExcludedAwareSiameseModel([1, 2, 3, 4, 5, 6, 7]),
    ExcludedAwareMnistCNNModel([8, 9, 0]),
    ExcludedAwareSiameseModel([8, 9, 0]),
    ExcludedAwareMnistCNNModel([5, 6, 7, 8, 9, 0]),
    ExcludedAwareSiameseModel([5, 6, 7, 8, 9, 0]),
])
