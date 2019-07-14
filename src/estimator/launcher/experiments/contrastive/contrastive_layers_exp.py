from abc import ABC
from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.contrastive_model import ContrastiveModel, MnistContrastiveModel, FmnistContrastiveModel, \
    ExtruderContrastiveModel
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.utils import consts


class ContrastiveLayersExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "contrastive_layers_exp"


class ConvParamsAwareContrastiveModel(ContrastiveModel, ABC):

    @property
    def summary(self) -> str:
        return self._summary_from_dict(
            {
                'f': self.filters,
                'ksl': self.kernel_side_lengths,
                'd': self.dense,
            }, stem=self.name[:1] + '_' + self.raw_data_provider.description.variant.name.lower()[:1])

    def __init__(self, filters, kernel_side_lengths, dense) -> None:
        super().__init__()
        self.filters = filters
        self.dense = dense
        self.kernel_side_lengths = kernel_side_lengths

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params,
                               {
                                   consts.FILTERS: self.filters,
                                   consts.KERNEL_SIDE_LENGTHS: self.kernel_side_lengths,
                                   consts.DENSE_UNITS: self.dense,
                               })


class MnistConvParamsAwareContrastiveModel(ConvParamsAwareContrastiveModel, MnistContrastiveModel):
    pass


class FmnistConvParamsAwareContrastiveModel(ConvParamsAwareContrastiveModel, FmnistContrastiveModel):
    pass


class ExtruderConvParamsAwareContrastiveModel(ConvParamsAwareContrastiveModel, ExtruderContrastiveModel):
    pass


launcher = ContrastiveLayersExperimentLauncher([
    # MnistConvParamsAwareContrastiveModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[7, 5, 3, 3, 3], dense=[]),
    # MnistConvParamsAwareContrastiveModel(filters=[32, 8], kernel_side_lengths=[5, 5], dense=[], concat_dense=[20, 2],
    #                                  dropout_rates=[0.5, None]),

    ######

    FmnistConvParamsAwareContrastiveModel(filters=[32, 64, 128, 256, 512, 512, 512],
                                          kernel_side_lengths=[7, 5, 5, 5, 5, 5, 5], dense=[256, 256, 2]),
    # FmnistConvParamsAwareContrastiveModel(filters=[32, 8], kernel_side_lengths=[5, 5]1, dense=[], concat_dense=[20, 2],
    #                                   dropout_rates=[0.5, None]),

    #####

    # ExtruderConvParamsAwareContrastiveModel(filters=[8, 16, 32, 64, 128, 320, 2], kernel_side_lengths=[5, 5, 5, 5, 5, 5, 5], dense=[])
    # ExtruderConvParamsAwareContrastiveModel(filters=[32, 8], kernel_side_lengths=[5, 5], dense=[], concat_dense=[20, 2],
    #                                     dropout_rates=[0.5, None])
])
