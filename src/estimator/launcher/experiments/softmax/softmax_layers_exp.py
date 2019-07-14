from abc import ABC
from typing import Dict, Any

from src.estimator.launcher.launchers import ExperimentLauncher
from src.estimator.model.estimator_conv_model import merge_two_dicts
from src.estimator.model.softmax_model import MnistSoftmaxModel, FmnistSoftmaxModel, ExtruderSoftmaxModel, SoftmaxModel
from src.utils import consts


class SoftmaxLayersExperimentLauncher(ExperimentLauncher):
    @property
    def name(self):
        return "softmax_layers_exp"


class ConvParamsAwareSoftmaxModel(SoftmaxModel, ABC):

    @property
    def summary(self) -> str:
        return self._summary_from_dict(
            {
                'f': self.filters,
                'ksl': self.kernel_side_lengths,
                'd': self.dense,
                'cd': self.concat_dense,
                'drop': len([x for x in self.dropout_rates if x])
            }, stem=self.name[:1] + '_' + self.raw_data_provider.description.variant.name.lower()[:1])

    def __init__(self, filters, kernel_side_lengths, dense, concat_dense, dropout_rates) -> None:
        super().__init__()
        self.filters = filters
        self.dropout_rates = dropout_rates
        self.concat_dense = concat_dense
        self.dense = dense
        self.kernel_side_lengths = kernel_side_lengths

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params,
                               {
                                   consts.FILTERS: self.filters,
                                   consts.KERNEL_SIDE_LENGTHS: self.kernel_side_lengths,
                                   consts.DENSE_UNITS: self.dense,
                                   consts.CONCAT_DENSE_UNITS: self.concat_dense,
                                   consts.CONCAT_DROPOUT_RATES: self.dropout_rates,
                               })


class MnistConvParamsAwareSoftmaxModel(ConvParamsAwareSoftmaxModel, MnistSoftmaxModel):
    pass


class FmnistConvParamsAwareSoftmaxModel(ConvParamsAwareSoftmaxModel, FmnistSoftmaxModel):
    pass


class ExtruderConvParamsAwareSoftmaxModel(ConvParamsAwareSoftmaxModel, ExtruderSoftmaxModel):
    pass


launcher = SoftmaxLayersExperimentLauncher([
    MnistConvParamsAwareSoftmaxModel(filters=[32, 64], kernel_side_lengths=[5, 5], dense=[], concat_dense=[100, 2],
                                     dropout_rates=[0.5, None]),
    # MnistConvParamsAwareSoftmaxModel(filters=[32, 8], kernel_side_lengths=[5, 5], dense=[], concat_dense=[20, 2],
    #                                  dropout_rates=[0.5, None]),

    ######

    FmnistConvParamsAwareSoftmaxModel(filters=[32, 64], kernel_side_lengths=[5, 5], dense=[], concat_dense=[100, 2],
                                      dropout_rates=[0.5, None]),
    # FmnistConvParamsAwareSoftmaxModel(filters=[32, 8], kernel_side_lengths=[5, 5], dense=[], concat_dense=[20, 2],
    #                                   dropout_rates=[0.5, None]),

    #####

    ExtruderConvParamsAwareSoftmaxModel(filters=[32, 64, 64], kernel_side_lengths=[5, 5, 5], dense=[],
                                        concat_dense=[100, 2],
                                        dropout_rates=[0.5, None]),
    # ExtruderConvParamsAwareSoftmaxModel(filters=[32, 8], kernel_side_lengths=[5, 5], dense=[], concat_dense=[20, 2],
    #                                     dropout_rates=[0.5, None])
])
