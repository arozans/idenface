import pytest
from mockito import ANY

from helpers import gen
from src.utils import filenames, before_run, utils


@pytest.fixture()
def rmtree_mock(mocker):
    return mocker.patch('shutil.rmtree')


def test_should_delete_existing_run_directory_for_default_launcher(when, rmtree_mock):
    run_data = gen.run_data(is_experiment=False)

    run_logs_dir = filenames.get_run_logs_data_dir(run_data)

    when(utils).check_filepath(filename=run_logs_dir, exists=True, is_directory=True, is_empty=ANY).thenReturn(True)

    before_run._prepare_dirs(None, run_data)
    rmtree_mock.assert_called_once_with(str(run_logs_dir))


def test_should_not_delete_existing_launcher_directory_for_default_launcher(when, rmtree_mock):
    run_data = gen.run_data(is_experiment=False, run_no=1)

    result = before_run._prepare_launcher_dir(run_data)

    assert result is None
    rmtree_mock.assert_not_called()


def test_should_delete_existing_launcher_directory_for_experiment_launcher_during_first_run(when, rmtree_mock):
    run_data = gen.run_data(is_experiment=True, run_no=1)
    launcher_dir = filenames.get_launcher_dir(run_data)
    when(utils).check_filepath(filename=launcher_dir, exists=True, is_directory=True, is_empty=ANY).thenReturn(True)

    result = before_run._prepare_launcher_dir(run_data)

    assert result == launcher_dir
    rmtree_mock.assert_called_once_with(str(launcher_dir))


def test_should_not_delete_existing_launcher_directory_for_experiment_launcher_when_its_not_first_run(when,
                                                                                                      rmtree_mock):
    run_data = gen.run_data(is_experiment=True, run_no=2)

    result = before_run._prepare_launcher_dir(run_data)

    assert result is None
    rmtree_mock.assert_not_called()
