from src.estimator.launcher.experiments.mixed.softmax_and_contrastive_single_excluded_exp import \
    ExcludedAwareMnistSoftmaxModel, ExcludedAwareMnistContrastiveModel
from src.estimator.launcher.launchers import ExperimentLauncher


class SoftmaxAndContrastiveMnistMultipleExcludedExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "softmax_and_contrastive_multiple_excluded_exp"


launcher = SoftmaxAndContrastiveMnistMultipleExcludedExperimentLauncher([
    ExcludedAwareMnistSoftmaxModel([1, 2]),
    ExcludedAwareMnistContrastiveModel([1, 2]),
    ExcludedAwareMnistSoftmaxModel([5, 6]),
    ExcludedAwareMnistContrastiveModel([5, 6]),
    ExcludedAwareMnistSoftmaxModel([1, 2, 3]),
    ExcludedAwareMnistContrastiveModel([1, 2, 3]),
    ExcludedAwareMnistSoftmaxModel([1, 2, 3, 4, 5]),
    ExcludedAwareMnistContrastiveModel([1, 2, 3, 4, 5]),
    ExcludedAwareMnistSoftmaxModel([1, 2, 3, 4, 5, 6, 7]),
    ExcludedAwareMnistContrastiveModel([1, 2, 3, 4, 5, 6, 7]),
    ExcludedAwareMnistSoftmaxModel([8, 9, 0]),
    ExcludedAwareMnistContrastiveModel([8, 9, 0]),
    ExcludedAwareMnistSoftmaxModel([5, 6, 7, 8, 9, 0]),
    ExcludedAwareMnistContrastiveModel([5, 6, 7, 8, 9, 0]),
    ExcludedAwareMnistSoftmaxModel([1, 2, 5, 6, 7, 8, 9, 0]),
    ExcludedAwareMnistContrastiveModel([1, 2, 5, 6, 7, 8, 9, 0]),
])
