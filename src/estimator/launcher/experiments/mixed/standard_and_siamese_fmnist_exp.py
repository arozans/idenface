from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
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
            consts.EXCLUDED_KEYS: [3, 4, 5, 6, 7, 8],
            consts.GLOBAL_SUFFIX: 'v4',
        }


class ConvParamsAwareFmnistSiameseModel(FmnistSiameseModel):

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


launcher = StandardAndSiameseFmnistExperimentLauncher([
    FmnistSiameseModel(),
    ConvParamsAwareFmnistSiameseModel(filters=[64, 128, 256, 512, 2], kernel_side_lengths=[9, 7, 5, 3, 3]),
    ConvParamsAwareFmnistSiameseModel(filters=[128, 256, 512, 1024, 2], kernel_side_lengths=[9, 7, 5, 3, 3]),
    ConvParamsAwareFmnistSiameseModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    ConvParamsAwareFmnistSiameseModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[2, 2, 2, 2, 2]),
    ConvParamsAwareFmnistSiameseModel(filters=[128, 256, 512, 1024, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    ConvParamsAwareFmnistSiameseModel(filters=[64, 128, 256, 512, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    ConvParamsAwareFmnistSiameseModel(filters=[512, 512, 512, 512, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    ConvParamsAwareFmnistSiameseModel(filters=[64, 64, 64, 64, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    ConvParamsAwareFmnistSiameseModel(filters=[32, 32, 32, 32, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    FmnistCNNModel()
])
