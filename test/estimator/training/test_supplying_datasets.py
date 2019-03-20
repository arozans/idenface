from pathlib import Path

import pytest

from helpers.test_helpers import FakeRawDataProvider, FakeMnistRawDataProvider
from helpers.tf_helpers import run_eagerly
from src.data.common_types import DatasetSpec, DatasetType
from src.estimator.training import supplying_datasets
from src.utils import consts
from src.utils.configuration import config

dataset_path = Path('foobar')
provider = FakeRawDataProvider


@pytest.fixture
def preparing_and_reading_mocks(mocker):
    preparing = mocker.patch('src.data.preparing_data.find_or_create_paired_data_dir', return_value=dataset_path,
                             autospec=True)
    reading = mocker.patch('src.data.saving.reading_tfrecords.assemble_dataset', autospec=True)
    return preparing, reading


def test_train_input_fn_should_search_for_dataset_with_correct_spec(preparing_and_reading_mocks):
    preparing, reading = preparing_and_reading_mocks

    supplying_datasets.train_input_fn(provider)
    preparing.assert_called_once_with(DatasetSpec(provider, DatasetType.TRAIN, with_excludes=False))
    reading.assert_called_once_with(dataset_path)


def test_eval_input_fn_not_ignoring_excludes_should_search_for_dataset_with_correct_spec(preparing_and_reading_mocks):
    preparing, reading = preparing_and_reading_mocks

    supplying_datasets.eval_input_fn(provider)
    preparing.assert_called_once_with(DatasetSpec(provider, DatasetType.TEST, with_excludes=False))
    reading.assert_called_once_with(dataset_path)


def test_eval_with_excludes_input_fn_should_search_for_dataset_with_correct_spec(preparing_and_reading_mocks):
    preparing, reading = preparing_and_reading_mocks

    supplying_datasets.eval_with_excludes_fn(provider)
    preparing.assert_called_once_with(DatasetSpec(provider, DatasetType.TEST, with_excludes=True))
    reading.assert_called_once_with(dataset_path)


@pytest.fixture
def patched_excluded(mocker):
    mocker.patch('src.utils.configuration._excluded_keys', [1, 2])


def get_class_name(clazz):
    return type(clazz()).__name__


def from_class_name(name: str):
    return eval(name.numpy())


@pytest.mark.parametrize('provider_cls_name',
                         # [get_class_name(MnistRawDataProvider)])
                         [get_class_name(FakeMnistRawDataProvider)])
@run_eagerly
def test_integration_train_input_fn(provider_cls_name):
    provider = from_class_name(provider_cls_name)
    side_len = provider.description().image_side_length
    dataset = supplying_datasets.train_input_fn(provider)
    iterator = dataset.make_one_shot_iterator()
    first_batch = iterator.get_next()

    left = first_batch[0][
        consts.LEFT_FEATURE_IMAGE].numpy()  # batch(1) => (side_len,side_len,1), batch(3) =>(3,side_len,side_len,1)
    right = first_batch[0][consts.RIGHT_FEATURE_IMAGE].numpy()
    assert left.shape == (config.batch_size, side_len, side_len, 1)
    assert right.shape == (config.batch_size, side_len, side_len, 1)
    label = first_batch[1].numpy()  # batch(1) => int, batch(3) =>(3,)
    assert label.shape == (config.batch_size,)
