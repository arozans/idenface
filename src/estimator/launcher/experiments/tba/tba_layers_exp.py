from abc import ABC
from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.tba_model import TBAModel, MnistTBAModel, FmnistTBAModel, ExtruderTBAModel
from src.utils import consts


class TBALayersExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "TBA_layers_exp"


class ConvParamsAwareTBAModel(TBAModel, ABC):

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


class MnistConvParamsAwareTBAModel(ConvParamsAwareTBAModel, MnistTBAModel):
    pass


class FmnistConvParamsAwareTBAModel(ConvParamsAwareTBAModel, FmnistTBAModel):
    pass


class ExtruderConvParamsAwareTBAModel(ConvParamsAwareTBAModel, ExtruderTBAModel):
    pass


launcher = TBALayersExperimentLauncher([
    # MnistConvParamsAwareTBAModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[7, 5, 3, 3, 3], dense=[]),
    # # MnistConvParamsAwareTBAModel(filters=[32, 8], kernel_side_lengths=[5, 5], dense=[], concat_dense=[20, 2],
    # #                                  dropout_rates=[0.5, None]),
    #
    # ######
    #
    # FmnistConvParamsAwareTBAModel(filters=[32, 64, 128, 256, 2], kernel_side_lengths=[7, 5, 3, 3, 3], dense=[]),
    # # FmnistConvParamsAwareTBAModel(filters=[32, 8], kernel_side_lengths=[5, 5], dense=[], concat_dense=[20, 2],
    # #                                   dropout_rates=[0.5, None]),

    #####
    ExtruderConvParamsAwareTBAModel(filters=[8, 16, 32, 64, 128, 256, 512], kernel_side_lengths=[5] * 7,
                                    dense=[10]),
    ExtruderConvParamsAwareTBAModel(filters=[8, 16, 32, 64, 128, 256, 512], kernel_side_lengths=[5] * 7,
                                    dense=[20]),
    ExtruderConvParamsAwareTBAModel(filters=[8, 16, 32, 64, 128, 256, 512], kernel_side_lengths=[5] * 7,
                                    dense=[30]),
    ExtruderConvParamsAwareTBAModel(filters=[8, 16, 32, 64, 128, 256, 512], kernel_side_lengths=[5] * 7,
                                    dense=[40]),
    ExtruderConvParamsAwareTBAModel(filters=[8, 16, 32, 64, 128, 256, 512], kernel_side_lengths=[5] * 7,
                                    dense=[50]),
    ExtruderConvParamsAwareTBAModel(filters=[8, 16, 32, 64, 128, 256, 512], kernel_side_lengths=[5] * 7,
                                    dense=[60]),
    ExtruderConvParamsAwareTBAModel(filters=[8, 16, 32, 64, 128, 256, 512], kernel_side_lengths=[5] * 7,
                                    dense=[70]),
    ExtruderConvParamsAwareTBAModel(filters=[8, 16, 32, 64, 128, 256, 512], kernel_side_lengths=[5] * 7,
                                    dense=[80])
])

# ExtruderConvParamsAwareTBAModel(filters=[32, 8], kernel_side_lengths=[5, `5], dense=[], concat_dense=[20, 2],
#                                     dropout_rates=[0.5, None])
