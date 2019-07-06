from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.contrastive_model import FmnistContrastiveModel
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.softmax_model import FmnistSoftmaxModel
from src.utils import consts


class SoftmaxAndContrastiveFmnistExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "softmax_and_contrastive_fmnist_exp"


class ConvParamsAwareFmnistContrastiveModel(FmnistContrastiveModel):

    @property
    def summary(self) -> str:
        return self.summary_from_dict(
            {
                "f": self.filters,
                "ksl": self.kernel_side_lengths
            })

    def __init__(self, filters, kernel_side_lengths) -> None:
        super().__init__()
        self.filters = filters
        self.kernel_side_lengths = kernel_side_lengths

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params,
                               {
                                   consts.FILTERS: self.filters,
                                   consts.KERNEL_SIDE_LENGTHS: self.kernel_side_lengths,
                               })


launcher = SoftmaxAndContrastiveFmnistExperimentLauncher([
    # ConvParamsAwareFmnistContrastiveModel(filters=[64, 128, 256, 512, 2], kernel_side_lengths=[9, 7, 5, 3, 3]),
    # ConvParamsAwareFmnistContrastiveModel(filters=[128, 256, 512, 1024, 2], kernel_side_lengths=[9, 7, 5, 3, 3]),
    # ConvParamsAwareFmnistContrastiveModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    # ConvParamsAwareFmnistContrastiveModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[2, 2, 2, 2, 2]),
    # ConvParamsAwareFmnistContrastiveModel(filters=[128, 256, 512, 1024, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    # ConvParamsAwareFmnistContrastiveModel(filters=[64, 128, 256, 512, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    # ConvParamsAwareFmnistContrastiveModel(filters=[512, 512, 512, 512, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    # ConvParamsAwareFmnistContrastiveModel(filters=[64, 64, 64, 64, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    # ConvParamsAwareFmnistContrastiveModel(filters=[32, 32, 32, 32, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    FmnistContrastiveModel(),
    FmnistSoftmaxModel()
])
