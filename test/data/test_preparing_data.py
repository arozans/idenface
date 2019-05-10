from pathlib import Path

import pytest

from src.data import preparing_data
from src.utils import filenames
from testing_utils import gen
from testing_utils.testing_classes import FAKE_TRAIN_DATASET_SPEC

PAIRS_DATASET_DIRECTORY_NAME = 'foobar'


def matcher_fn(str_to_check):
    return str_to_check == PAIRS_DATASET_DIRECTORY_NAME


@pytest.fixture()
def prepare_mocks(mocker):
    mocker.patch('src.utils.filenames.create_dataset_directory_name', return_value=PAIRS_DATASET_DIRECTORY_NAME,
                 autospec=True)
    creating_dataset_mock = mocker.patch('src.data.processing.creating_paired_data.create_paired_data', autospec=True,
                                         return_value=('foo', 'bar'))
    matcher_fn_mock = mocker.patch('src.data.preparing_data.get_dataset_dir_matcher_fn', return_value=matcher_fn,
                                   autospec=True)
    save_to_tfrecord_mock = mocker.patch('src.data.tfrecord.saving.saving_tfrecords.save_to_tfrecord', autospec=True)

    return creating_dataset_mock, matcher_fn_mock, save_to_tfrecord_mock


def prepare_pairs_dataset_directory(create_content):
    tmp_foo_bar_dir = (filenames.get_processed_input_data_dir(gen.dataset_spec()) / PAIRS_DATASET_DIRECTORY_NAME)
    tmp_foo_bar_dir.mkdir(exist_ok=True, parents=True)
    if create_content:
        (tmp_foo_bar_dir / 'foo.txt').write_text('Some text')
    return tmp_foo_bar_dir


def test_should_return_correct_not_empty_dir_with_dataset(prepare_mocks):
    creating_dataset_mock, matcher_fn_mock, save_to_tfrecord_mock = prepare_mocks
    tmp_foo_bar_dir = prepare_pairs_dataset_directory(create_content=True)

    dataset_dir = preparing_data.find_or_create_dataset_dir(FAKE_TRAIN_DATASET_SPEC)

    assert dataset_dir == tmp_foo_bar_dir
    matcher_fn_mock.assert_called_once()
    creating_dataset_mock.assert_not_called()
    save_to_tfrecord_mock.assert_not_called()


def test_should_create_dataset_when_dir_is_empty(prepare_mocks):
    creating_dataset_mock, matcher_fn_mock, save_to_tfrecord_mock = prepare_mocks
    tmp_foo_bar_dir = prepare_pairs_dataset_directory(create_content=False)

    dataset_dir = preparing_data.find_or_create_dataset_dir(FAKE_TRAIN_DATASET_SPEC)

    assert dataset_dir == tmp_foo_bar_dir
    matcher_fn_mock.assert_called_once_with(FAKE_TRAIN_DATASET_SPEC)
    creating_dataset_mock.assert_called_once_with(FAKE_TRAIN_DATASET_SPEC)
    save_to_tfrecord_mock.assert_called_once()


def test_should_create_dataset_when_dir_doesnt_exist(prepare_mocks):
    creating_dataset_mock, matcher_fn_mock, save_to_tfrecord_mock = prepare_mocks

    dataset_dir = preparing_data.find_or_create_dataset_dir(FAKE_TRAIN_DATASET_SPEC)

    assert dataset_dir == Path(
        filenames.get_processed_input_data_dir(gen.dataset_spec())) / PAIRS_DATASET_DIRECTORY_NAME
    assert dataset_dir.exists()
    matcher_fn_mock.assert_not_called()
    creating_dataset_mock.assert_called_once_with(FAKE_TRAIN_DATASET_SPEC)
    save_to_tfrecord_mock.assert_called_once()
