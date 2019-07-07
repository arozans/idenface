from typing import Dict, Any

from src.data.common_types import AbstractRawDataProvider
from src.data.raw_data.raw_data_providers import ExtruderRawDataProvider
from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.contrastive_model import ContrastiveModel
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.utils import consts


class ExtruderContrastiveExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "contrastive_extruder_exp"


class ConvParamsAwareExtruderContrastiveModel(ContrastiveModel):

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return ExtruderRawDataProvider(100)

    @property
    def summary(self) -> str:
        return self._summary_from_dict(
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
                                   consts.TRAIN_STEPS: 5000,
                                   consts.BATCH_SIZE: 300,
                               })


launcher = ExtruderContrastiveExperimentLauncher([
    ConvParamsAwareExtruderContrastiveModel(filters=[32, 2], kernel_side_lengths=[3, 3]),
    ConvParamsAwareExtruderContrastiveModel(filters=[32, 2], kernel_side_lengths=[5, 5]),
    ConvParamsAwareExtruderContrastiveModel(filters=[32, 32, 2], kernel_side_lengths=[3, 3, 3]),  # useless
    ConvParamsAwareExtruderContrastiveModel(filters=[32, 32, 2], kernel_side_lengths=[5, 5, 5]),
    ConvParamsAwareExtruderContrastiveModel(filters=[64, 64, 2], kernel_side_lengths=[5, 5, 5]),
    ConvParamsAwareExtruderContrastiveModel(filters=[8, 16, 32, 64, 128, 320, 80],
                                            kernel_side_lengths=[3, 3, 3, 3, 3, 3, 3]),
    ConvParamsAwareExtruderContrastiveModel(filters=[8, 16, 32, 64, 128, 320, 80],
                                            kernel_side_lengths=[5, 5, 5, 5, 5, 5, 5]),
    ConvParamsAwareExtruderContrastiveModel(filters=[8, 16, 32, 64, 128, 320, 2],
                                            kernel_side_lengths=[3, 3, 3, 3, 3, 3, 3]),
    ConvParamsAwareExtruderContrastiveModel(filters=[8, 16, 32, 64, 128, 320, 2],
                                            kernel_side_lengths=[5, 5, 5, 5, 5, 5, 5]),  # the best one
    ConvParamsAwareExtruderContrastiveModel(filters=[32, 32, 32, 32, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    ConvParamsAwareExtruderContrastiveModel(filters=[32, 32, 32, 32, 2], kernel_side_lengths=[5, 5, 5, 5, 5]),
    ConvParamsAwareExtruderContrastiveModel(filters=[64, 64, 64, 64, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    ConvParamsAwareExtruderContrastiveModel(filters=[64, 64, 64, 64, 2], kernel_side_lengths=[5, 5, 5, 5, 5]),
    ConvParamsAwareExtruderContrastiveModel(filters=[64, 64, 64, 64, 2], kernel_side_lengths=[7, 7, 7, 7, 7]),
    ConvParamsAwareExtruderContrastiveModel(filters=[64, 128, 256, 512, 2], kernel_side_lengths=[9, 7, 5, 3, 3]),
    ConvParamsAwareExtruderContrastiveModel(filters=[128, 256, 512, 1024, 2], kernel_side_lengths=[9, 7, 5, 3, 3]),
    # ConvParamsAwareExtruderContrastiveModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    # ConvParamsAwareExtruderContrastiveModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[2, 2, 2, 2, 2]),
    # ConvParamsAwareExtruderContrastiveModel(filters=[128, 256, 512, 1024, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    # ConvParamsAwareExtruderContrastiveModel(filters=[64, 128, 256, 512, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),
    # ConvParamsAwareExtruderContrastiveModel(filters=[512, 512, 512, 512, 2], kernel_side_lengths=[3, 3, 3, 3, 3]),

])
