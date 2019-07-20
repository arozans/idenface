from abc import ABC

import pytest
from hamcrest import assert_that, has_entries

from src.data.common_types import AbstractRawDataProvider, ImageDimensions
from src.estimator.model.contrastive_model import ContrastiveModel
from src.estimator.model.estimator_conv_model import EstimatorConvModel
from src.estimator.model.softmax_model import SoftmaxModel
from src.estimator.model.tba_model import TBAModel
from src.utils import consts
from testing_utils import gen
from testing_utils.testing_classes import FakeRawDataProvider


class _FakeImageDimensionParametrizedModel(EstimatorConvModel, ABC):

    def __init__(self, dimensions: ImageDimensions):
        self.description = gen.dataset_desc(image_dimensions=dimensions)

    @property
    def raw_data_provider(self) -> AbstractRawDataProvider:
        return FakeRawDataProvider(description=self.description)


class _FakeSoftmaxModel(_FakeImageDimensionParametrizedModel, SoftmaxModel):
    pass


class _FakeContrastiveModel(_FakeImageDimensionParametrizedModel, ContrastiveModel):
    pass


class _FakeTBAModel(_FakeImageDimensionParametrizedModel, TBAModel):
    pass


@pytest.mark.parametrize('model', [
    _FakeSoftmaxModel,
])
@pytest.mark.parametrize('patched_params, image_dims, expected_param_nums', [
    ({
         consts.FILTERS: [32, 8],
         consts.KERNEL_SIDE_LENGTHS: [5, 5],
         consts.DENSE_UNITS: [],
         consts.CONCAT_DENSE_UNITS: [4, 2],
         consts.CONCAT_DROPOUT_RATES: [0.5, None]
     },
     ImageDimensions(200, 200, 1),
     {
         consts.ALL_PARAMS_COUNT: 174494,
         consts.CONV_PARAMS_COUNT: 7240,
         consts.DENSE_PARAMS_COUNT: 0,
         consts.CONCAT_DENSE_PARAMS_COUNT: 160014
     }),
    ({
         consts.FILTERS: [32, 8],
         consts.KERNEL_SIDE_LENGTHS: [5, 5],
         consts.DENSE_UNITS: [],
         consts.CONCAT_DENSE_UNITS: [4, 2],
         consts.CONCAT_DROPOUT_RATES: [0.5, None]
     },
     ImageDimensions(100, 100, 3),
     {
         consts.ALL_PARAMS_COUNT: 57694,
         consts.CONV_PARAMS_COUNT: 8840,
         consts.DENSE_PARAMS_COUNT: 0,
         consts.CONCAT_DENSE_PARAMS_COUNT: 40014

     }),
    ({
         consts.FILTERS: [32, 8],
         consts.KERNEL_SIDE_LENGTHS: [5, 5],
         consts.DENSE_UNITS: [],
         consts.CONCAT_DENSE_UNITS: [4, 2],
         consts.CONCAT_DROPOUT_RATES: [0.5, None]
     },
     ImageDimensions(28, 28, 1),
     {
         consts.ALL_PARAMS_COUNT: 17630,
         consts.CONV_PARAMS_COUNT: 7240,
         consts.DENSE_PARAMS_COUNT: 0,
         consts.CONCAT_DENSE_PARAMS_COUNT: 3150

     })
], indirect=['patched_params'])
def test_should_correctly_calculate_softmax_network_parameters(model, patched_params, image_dims, expected_param_nums):
    result = model(image_dims).get_parameters_count_dict()

    assert len(result) == 5
    assert_that(result, has_entries(expected_param_nums))


@pytest.mark.parametrize('model', [
    _FakeContrastiveModel,
    _FakeTBAModel,
])
@pytest.mark.parametrize('patched_params, image_dims, expected_param_nums', [

    ({
         consts.FILTERS: [32, 64, 128, 256],
         consts.KERNEL_SIDE_LENGTHS: [7, 5, 3, 1],
         consts.DENSE_UNITS: [30, 2]
     },
     ImageDimensions(200, 200, 1),
     {
         consts.CONV_PARAMS_COUNT: 159744,
         consts.DENSE_PARAMS_COUNT: 1298012,
         consts.ALL_PARAMS_COUNT: 1457756,
     }),
    ({
         consts.FILTERS: [32, 64, 128, 256],
         consts.KERNEL_SIDE_LENGTHS: [7, 5, 3, 1],
         consts.DENSE_UNITS: [30, 2]
     },
     ImageDimensions(28, 28, 1),
     {
         consts.CONV_PARAMS_COUNT: 159744,
         consts.DENSE_PARAMS_COUNT: 30812,
         consts.ALL_PARAMS_COUNT: 190556,
     }),
    ({
         consts.FILTERS: [8, 16, 32, 64, 128, 320, 2],
         consts.KERNEL_SIDE_LENGTHS: [5, 5, 5, 5, 5, 5, 5],
         consts.DENSE_UNITS: [30, 2]
     },
     ImageDimensions(100, 100, 3),
     {
         consts.CONV_PARAMS_COUNT: 1313170,
         consts.DENSE_PARAMS_COUNT: 152,
         consts.ALL_PARAMS_COUNT: 1313322,
     })

], indirect=['patched_params'])
def test_should_correctly_calculate_siamese_network_parameters(model, patched_params, image_dims, expected_param_nums):
    result = model(image_dims).get_parameters_count_dict()

    assert len(result) == 4
    assert_that(result, has_entries(expected_param_nums))
    assert consts.CONCAT_DENSE_PARAMS_COUNT not in result.keys()
