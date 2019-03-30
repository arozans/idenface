from pathlib import Path

import pytest
from hamcrest import assert_that, starts_with, ends_with

from helpers import gen
from helpers.fake_estimator_model import FakeModel
from helpers.test_consts import MNIST_TRAIN_DATASET_SPEC_IGNORING_EXCLUDES, \
    FAKE_TRAIN_DATASET_SPEC, FAKE_TEST_DATASET_SPEC, \
    MNIST_TEST_DATASET_SPEC_IGNORING_EXCLUDES, MNIST_TRAIN_DATASET_SPEC
from src.estimator.launcher.launchers import DefaultLauncher, RunData
from src.utils import filenames, consts


def test_home_directory(unpatched_home_dir):
    dir = filenames._get_home_directory()

    assert_that(str(dir), starts_with('/home/'))


def test_home_tf_directory(unpatched_home_dir):
    dir = filenames._get_home_tf_directory()

    assert_that(str(dir), starts_with('/home/'))
    assert_that(str(dir), ends_with('/tf'))


def test_raw_input_data_dirs_placement():
    raw_input_data_dir = filenames.get_raw_input_data_dir()

    assert_that(str(raw_input_data_dir), ends_with('/tf/datasets/raw'))


@pytest.mark.parametrize('encoding', [False, True])
def test_processed_input_data_dirs_placement(encoding):
    processed_input_data_dir = filenames.get_processed_input_data_dir(encoding)

    expected = '/tf/datasets/' + ((consts.NOT_ENCODED_DIR_FRAGMENT + '/') if not encoding else '') + 'paired'
    assert_that(str(processed_input_data_dir), ends_with(expected))


FILENAME_DATASET_CONFIG_EXCLUDED_PARAMETERS = [
    ('foo_train', FAKE_TRAIN_DATASET_SPEC, []),
    ('mnist_train', MNIST_TRAIN_DATASET_SPEC, []),
    ('foo_train_ex_3-2-4', FAKE_TRAIN_DATASET_SPEC, [3, 2, 4]),
    ('mnist_train', MNIST_TRAIN_DATASET_SPEC_IGNORING_EXCLUDES, [3, 2, 4]),
    ('foo_test', FAKE_TEST_DATASET_SPEC, []),
    ('foo_test_ex_1-2-3', FAKE_TEST_DATASET_SPEC, [1, 2, 3]),
    ('mnist_test', MNIST_TEST_DATASET_SPEC_IGNORING_EXCLUDES, []),
    ('mnist_test', MNIST_TEST_DATASET_SPEC_IGNORING_EXCLUDES, [1, 2, 3]),
]


@pytest.mark.parametrize('correct_name, dataset_config, excluded', FILENAME_DATASET_CONFIG_EXCLUDED_PARAMETERS)
def test_should_create_correct_dataset_dir_and_tfrecords_name(correct_name, dataset_config, excluded, mocker):
    mocker.patch('src.utils.configuration._excluded_keys', excluded)
    result = 'd180711t022345'
    mocker.patch('time.strftime', return_value=result)
    dir_name = filenames.create_pairs_dataset_directory_name(dataset_config)
    assert dir_name == correct_name


def test_should_create_run_dir_using_launcher(mocker):
    launcher = DefaultLauncher([FakeModel()])
    summary = "run_summary"
    mocker.patch('src.estimator.launcher.providing_launcher.provide_launcher', return_value=launcher)
    get_run_summary_mock = mocker.patch('src.utils.utils.get_run_summary', return_value=summary)
    run_data: RunData = launcher.runs_data[0]

    dir_name = filenames.get_run_dir(run_data)

    assert_that(str(dir_name), ends_with(
        '/tf/runs/{}/{}/{}'.format(run_data.runs_directory_name, run_data.launcher_name, summary)))
    get_run_summary_mock.assert_called_once_with(run_data.model.summary)


@pytest.fixture
def run_dir_mock(mocker):
    mocker.patch('src.utils.filenames.get_run_dir', return_value=Path('baz'))


def test_run_logging_dir(run_dir_mock):
    run_logging_dir = filenames.get_run_logs_data_dir(gen.run_data())

    assert_that(str(run_logging_dir), ends_with('baz/logs'))


def test_run_text_logs_dir(run_dir_mock):
    run_logging_dir = filenames.get_run_text_logs_dir(gen.run_data())

    assert_that(str(run_logging_dir), ends_with('baz/text_logs'))


mocked_strftime = 'd180711t012345'


def test_should_create_correct_text_log_name(mocker, run_dir_mock):
    mocker.patch('time.strftime', return_value=mocked_strftime)
    summary = "summary"
    get_run_summary_mock = mocker.patch('src.utils.utils.get_run_summary', return_value=summary)

    model = FakeModel()
    dir_name = filenames.create_text_log_name(model)
    assert_that(str(dir_name), ends_with(summary + '_' + mocked_strftime + '.log'))
    get_run_summary_mock.assert_called_once_with(model.summary)
