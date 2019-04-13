import pytest
from dataclasses import replace

from src.data.raw_data import raw_data
from testing_utils.testing_classes import FAKE_TRAIN_DATASET_SPEC


@pytest.fixture
def dataset_spec_with_mock(mocker):
    return replace(FAKE_TRAIN_DATASET_SPEC, raw_data_provider_cls=mocker.Mock())


def test_should_return_train_dataset_when_requested(dataset_spec_with_mock):
    raw_data.get_raw_data(dataset_spec_with_mock)
    provider_mock = dataset_spec_with_mock.raw_data_provider_cls()
    assert provider_mock.get_raw_train.called
    assert provider_mock.get_raw_test.not_called


def test_should_return_test_dataset_when_requested(dataset_spec_with_mock):
    raw_data.get_raw_data(dataset_spec_with_mock)
    provider_mock = dataset_spec_with_mock.raw_data_provider_cls()
    assert provider_mock.get_raw_train.called
    assert provider_mock.get_raw_test.not_called
