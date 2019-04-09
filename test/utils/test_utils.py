import pytest

from estimator.training.integration.test_integration_training import FakeExperimentLauncher
from helpers.fake_estimator_model import FakeModel
from src.utils import utils

FILENAME_SUMMARY_EXCLUDED_PARAMETERS = [
    ('', None, []),
    ('_foo', 'foo', []),
    ('_ex_1-2', None, [1, 2]),
    ('_loops_200_ex_1-2-123', 'loops_200', [1, 2, 123]),
]


@pytest.mark.parametrize('suffix, global_suffix, excluded', FILENAME_SUMMARY_EXCLUDED_PARAMETERS)
def test_should_create_correct_run_summary(mocker, suffix, global_suffix, excluded):
    model = FakeModel()
    mocker.patch('src.utils.configuration._global_suffix', global_suffix)
    mocker.patch('src.utils.configuration._excluded_keys', excluded)

    dir_name = utils.get_run_summary(model)
    assert dir_name.endswith(suffix)


def test_check_directory_or_file(patched_home_dir):
    non_existing = patched_home_dir / 'not_exists'
    assert utils.check_filepath(non_existing, exists=False)
    assert not utils.check_filepath(non_existing, exists=True)
    assert utils.check_filepath(non_existing, exists=False, is_directory=True, is_empty=True)

    empty_dir = (patched_home_dir / 'baz')
    empty_dir.mkdir()
    assert not utils.check_filepath(empty_dir, exists=False)
    assert not utils.check_filepath(empty_dir, is_directory=False)
    assert utils.check_filepath(empty_dir, exists=True, is_directory=True, is_empty=True)
    assert not utils.check_filepath(empty_dir, exists=True, is_directory=True, is_empty=False)

    nonempty_dir = (patched_home_dir / 'qux')
    (nonempty_dir / 'quux').mkdir(parents=True)
    assert not utils.check_filepath(nonempty_dir, exists=False)
    assert not utils.check_filepath(nonempty_dir, is_directory=False)
    assert utils.check_filepath(nonempty_dir, exists=True, is_directory=True, is_empty=False)
    assert not utils.check_filepath(nonempty_dir, exists=True, is_directory=True, is_empty=True)

    empty_file = (patched_home_dir / 'abc.txt')
    empty_file.write_text('')
    assert not utils.check_filepath(empty_file, exists=False)
    assert not utils.check_filepath(empty_file, is_directory=True)
    assert utils.check_filepath(empty_file, exists=True, is_directory=False, is_empty=True)
    assert not utils.check_filepath(empty_file, exists=True, is_directory=False, is_empty=False)

    non_empty_file = (patched_home_dir / 'qwop.txt')
    non_empty_file.write_text('wubba lubba dub dub')
    assert not utils.check_filepath(non_empty_file, exists=False)
    assert not utils.check_filepath(non_empty_file, is_directory=True)
    assert utils.check_filepath(non_empty_file, exists=True, is_directory=False, is_empty=False)
    assert not utils.check_filepath(non_empty_file, exists=True, is_directory=False, is_empty=True)


def test_should_return_user_run_selection(mocker):
    model = FakeModel()
    launcher = FakeExperimentLauncher([
        FakeModel(),
        FakeModel(),
        model,
        FakeModel()
    ])
    mocker.patch('builtins.input', return_value='2')
    assert utils.user_run_selection(launcher).model == model
