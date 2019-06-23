from typing import Dict, Any

from src.data.common_types import AbstractRawDataProvider
from src.data.raw_data.raw_data_providers import ExtruderRawDataProvider
from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_model import merge_two_dicts
from src.estimator.model.siamese_conv_model import MnistSiameseModel
from src.utils import consts


class StandardExtruderExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "siamese_extruder_exp"


class ConvParamsAwareExtruderSiameseModel(MnistSiameseModel):

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return ExtruderRawDataProvider(100)

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
                                   consts.TRAIN_STEPS: 2000
                               })


launcher = StandardExtruderExperimentLauncher([
    # ConvParamsAwareExtruderSiameseModel(filters=[32, 2], kernel_side_lengths=[3, 3]),
    # ConvParamsAwareExtruderSiameseModel(filters=[32, 2], kernel_side_lengths=[5, 5]),
    # ConvParamsAwareExtruderSiameseModel(filters=[32, 32, 2], kernel_side_lengths=[3, 3, 3]), #check it
    # ConvParamsAwareExtruderSiameseModel(filters=[32, 32, 2], kernel_side_lengths=[5, 5, 5]), #check it
    # ConvParamsAwareExtruderSiameseModel(filters=[64, 64, 2], kernel_side_lengths=[5, 5, 5]), #check it
    ConvParamsAwareExtruderSiameseModel(filters=[32, 32, 32, 32, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    ConvParamsAwareExtruderSiameseModel(filters=[32, 32, 32, 32, 2], kernel_side_lengths=[5, 5, 5, 5, 5]),
    ConvParamsAwareExtruderSiameseModel(filters=[64, 64, 64, 64, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    ConvParamsAwareExtruderSiameseModel(filters=[64, 64, 64, 64, 2], kernel_side_lengths=[5, 5, 5, 5, 5]),
    ConvParamsAwareExtruderSiameseModel(filters=[64, 64, 64, 64, 2], kernel_side_lengths=[7, 7, 7, 7, 7]),
    ConvParamsAwareExtruderSiameseModel(filters=[64, 128, 256, 512, 2], kernel_side_lengths=[9, 7, 5, 3, 3]),
    ConvParamsAwareExtruderSiameseModel(filters=[128, 256, 512, 1024, 2], kernel_side_lengths=[9, 7, 5, 3, 3]),
    # ConvParamsAwareExtruderSiameseModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    # ConvParamsAwareExtruderSiameseModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[2, 2, 2, 2, 2]),
    # ConvParamsAwareExtruderSiameseModel(filters=[128, 256, 512, 1024, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    # ConvParamsAwareExtruderSiameseModel(filters=[64, 128, 256, 512, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    # ConvParamsAwareExtruderSiameseModel(filters=[512, 512, 512, 512, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),

])
