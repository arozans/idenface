from typing import Dict, Any, Type

import numpy as np
import pytest
from hamcrest import assert_that, has_entries

from data.conftest import NumberTranslationRawDataProvider
from src.data.common_types import AbstractRawDataProvider
from src.estimator.model.estimator_model import EstimatorModel, merge_two_dicts
from src.utils import consts
from testing_utils.testing_classes import FakeRawDataProvider, FakeModel

base_model_params_count = 2


class _BaseModel(EstimatorModel):

    def get_predicted_labels(self, result: np.ndarray):
        pass

    def get_predicted_scores(self, result: np.ndarray):
        pass

    @property
    def raw_data_provider_cls(self) -> Type[AbstractRawDataProvider]:  # todo remove this
        return FakeRawDataProvider

    @property
    def name(self) -> str:
        return "Base"

    def get_model_fn(self):
        pass

    def summary(self):
        pass

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {})


class _FirstInhModel(_BaseModel):

    @property
    def raw_data_provider_cls(self) -> Type[AbstractRawDataProvider]:
        return NumberTranslationRawDataProvider

    @property
    def name(self) -> str:
        return "Extended"

    def get_model_fn(self):
        pass

    def summary(self):
        pass

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {
            'foo': 1,
            'bar': 11,
        })


class _SecondInhModel(_FirstInhModel):

    @property
    def raw_data_provider_cls(self) -> Type[AbstractRawDataProvider]:
        return FakeRawDataProvider

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {
            'bar': 10,
            'baz': 100,
        })


def test_base_estimator_model_should_have_only_data_providers():
    model = _BaseModel()
    assert len(model.params) == base_model_params_count
    assert_that(model.params, has_entries({
        consts.DATASET_PROVIDER_CLS: model.dataset_provider_cls,
        consts.RAW_DATA_PROVIDER_CLS: model.raw_data_provider_cls,
    }))


def test_base_estimator_model_should_throw_on_non_existing_params():
    model = _BaseModel()
    with pytest.raises(KeyError):
        foo = model.params['some_fancy_param']


def test_first_inheritor_should_add_params():
    model = _FirstInhModel()
    assert len(model.params) == base_model_params_count + 2
    assert_that(model.params, has_entries({
        consts.RAW_DATA_PROVIDER_CLS: NumberTranslationRawDataProvider,
        'foo': 1,
        'bar': 11,
    }))


def test_second_inheritor_should_add_params():
    model = _SecondInhModel()
    assert len(model.params) == base_model_params_count + 3
    assert_that(model.params, has_entries({
        consts.RAW_DATA_PROVIDER_CLS: FakeRawDataProvider,
        'foo': 1,
        'bar': 10,
        'baz': 100,
    }))


class _ThirdInhModel(_SecondInhModel):
    pass


class _FourthInhModel(_ThirdInhModel):

    @property
    def additional_model_params(self) -> Dict[str, Any]:
        return merge_two_dicts(super().additional_model_params, {
            'qux': 1000,
        })


def test_third_inheritor_should_not_change_params():
    model = _ThirdInhModel()
    assert len(model.params) == base_model_params_count + 3
    assert_that(model.params, has_entries({
        consts.RAW_DATA_PROVIDER_CLS: FakeRawDataProvider,
        'foo': 1,
        'bar': 10,
        'baz': 100,
    }))


def test_fourth_inheritor_should_add_params():
    model = _FourthInhModel()
    assert len(model.params) == base_model_params_count + 4
    assert_that(model.params, has_entries({
        consts.RAW_DATA_PROVIDER_CLS: FakeRawDataProvider,
        'foo': 1,
        'bar': 10,
        'baz': 100,
        'qux': 1000,
    }))


@pytest.mark.parametrize('summary_dict, expected_result_ending', [
    ({"abc": 123, "def": False}, "_abc_123_def_False"),
    ({"aaa": [1, 2, 3], "BBB": None}, "_aaa_1_2_3_BBB_None"),
    ({"QWERTY": ["a", "b", "c"]}, "_QWERTY_a_b_c"),
    ({}, ""),
])
def test_should_create_summary_from_dict(summary_dict, expected_result_ending):
    model = FakeModel()
    res = model.summary_from_dict(summary_dict)
    assert res == model.name + expected_result_ending
