import pytest
import tensorflow as tf
from hamcrest import assert_that, has_entries, has_entry, not_
from tensorflow import flags as tf_flags

import helpers.test_helpers
from helpers.fake_estimator_model import FakeModel
from src.estimator.launcher.launchers import DefaultLauncher
from src.utils import configuration, filenames
from src.utils.configuration import config

PARAM_NAME = 'unknown_name'


@pytest.fixture(autouse=True)
def set_up(mocker):
    try:
        for name in list(tf_flags.FLAGS):
            delattr(tf_flags.FLAGS, name)
    except AttributeError:
        pass
    mocker.patch('sys.exit')


def pass_cli_arg(cl_flags):
    import sys
    sys.argv = sys.argv[0:2]
    [sys.argv.append('--' + str(f[0]) + '=' + str(f[1])) for f in cl_flags]


def define_command_line_args(flag_names):
    [tf_flags.DEFINE_integer(flag, None, 'no-help :(') for flag in flag_names]


def set_file_param(name, value):
    setattr(configuration, '_' + name, value)


def set_model_param(name, value):
    config.set_model_params({name: value})


def assert_param(param_name, expected_value):
    def main(*args):
        assert getattr(config, param_name) == expected_value

    tf.app.run(main)


def test_should_return_none_when_no_args_passed():
    found_flag = getattr(config, PARAM_NAME)

    assert found_flag is None


def test_should_return_consts_param():
    set_file_param(PARAM_NAME, 45)

    found_flag = getattr(config, PARAM_NAME)

    assert found_flag == 45


def test_should_return_param_when_passed_as_cl_arg():
    define_command_line_args([PARAM_NAME])
    pass_cli_arg([(PARAM_NAME, 42)])

    assert_param(PARAM_NAME, 42)


def test_should_prioritize_cl_args_over_model_params():
    set_model_param(PARAM_NAME, 45)
    define_command_line_args([PARAM_NAME])
    pass_cli_arg([(PARAM_NAME, 42)])

    assert_param(PARAM_NAME, 42)


def test_should_prioritize_cl_args_over_file_params():
    setattr(config, '_' + PARAM_NAME, 45)
    define_command_line_args([PARAM_NAME])
    pass_cli_arg([(PARAM_NAME, 42)])

    assert_param(PARAM_NAME, 42)


def test_should_prioritize_model_params_over_file_params():
    setattr(config, '_' + PARAM_NAME, 45)
    set_model_param(PARAM_NAME, 42)

    assert_param(PARAM_NAME, 42)


def test_should_return_none_when_arg_not_defined():
    not_defined_arg_name = 'foo'

    pass_cli_arg([(not_defined_arg_name, 42)])
    assert_param(not_defined_arg_name, None)


def test_should_log_when_flag_not_defined_during_training(mocker):
    launcher = DefaultLauncher([(FakeModel())])
    mocker.patch('src.utils.image_summaries.create_pair_summaries')
    mocker.patch('src.estimator.training.training.train')
    mocker.patch('src.estimator.launcher.providing_launcher.provide_launcher',
                 return_value=launcher)

    undefined_commandline_flag = ("foo", 34)
    defined_commandline_flag = ("batch_size", 42)
    pass_cli_arg([undefined_commandline_flag, defined_commandline_flag])

    helpers.test_helpers.run_app()

    log = list(filenames.get_run_text_logs_dir(launcher.runs_data[0]).iterdir())[0].read_text()
    assert "Undefined commandline flags: ['--foo=34']" in log


def test_should_get_multiple_file_params():
    set_file_param('uno', 1)
    set_file_param('dos', 2)
    set_file_param('tres', 3)

    file_params = configuration.get_file_params()
    assert_that(file_params, has_entries({'uno': 1, 'dos': 2, 'tres': 3}))


def test_should_get_file_params():
    set_file_param('uno', 1)
    pass_cli_arg([(PARAM_NAME, 42)])

    def check(*args):
        params = configuration.get_file_params()
        assert_that(params, has_entry('uno', 1))
        assert_that(params, not_(has_entry(PARAM_NAME, 42)))

    tf.app.run(check)


def test_should_get_cli_args():
    define_command_line_args(['dos', 'tres'])
    set_file_param('uno', 1)
    pass_cli_arg([('dos', 2), ('tres', 3)])

    def check(*args):
        params = configuration.get_commandline_flags()
        assert_that(params, has_entries({'dos': 2, 'tres': 3}))
        assert_that(params, not_(has_entry('uno', 1)))

    tf.app.run(check)


def test_should_get_model_params():
    define_command_line_args(['uno'])
    pass_cli_arg([('uno', 2)])
    set_file_param('dos', 2)
    set_model_param('tres', 3)

    def check(*args):
        model_params = configuration.get_model_params()
        assert_that(model_params, has_entries({'tres': 3}))
        assert_that(model_params, not_(has_entries({'uno': 1, 'dos': 2})))

    tf.app.run(check)


def test_check_defining_cli_args():
    if not tf_flags.FLAGS.find_module_defining_flag('batch_size'):
        configuration.define_cli_args()
    cl_flags = [
        ('batch_size', 12),
        ('optimizer', 'Adam'),
        ('learning_rate', 0.45),
        ('train_steps', 12000),
        ('eval_steps_interval', 123),
        ('excluded_keys', '1, 2, 3, foobar')  # pass list without brackets
    ]
    pass_cli_arg(cl_flags)

    def main(*args):
        for flag in cl_flags:
            found_flag = getattr(config, flag[0])
            if flag[0] != 'excluded_keys':
                assert found_flag == flag[1]
            else:
                assert found_flag == ['1', '2', '3', 'foobar']

    tf.app.run(main)
    tf_flags.FLAGS.unparse_flags()
