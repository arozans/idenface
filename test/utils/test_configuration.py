import copy

import pytest
import tensorflow as tf
from hamcrest import assert_that, has_entries, has_entry, not_
from tensorflow import flags as tf_flags

import testing_utils.testing_helpers
from src.estimator.launcher.launchers import DefaultLauncher
from src.utils import configuration, filenames, consts
from testing_utils.testing_classes import FakeModel

PATCHED_CONFIGURATION_VALUES = [
    (consts.BATCH_SIZE, 12),
    (consts.OPTIMIZER, 'Adam'),
    (consts.LEARNING_RATE, 0.45),
    (consts.TRAIN_STEPS, 12000),
    (consts.EVAL_STEPS_INTERVAL, 123),
    (consts.EXCLUDED_KEYS, '1, 2, 3, foobar')  # pass list without brackets
]

PARAM_NAME = 'param_name'

_config = None
config = None


@pytest.fixture(autouse=True, scope="module")
def before_module():
    from src.utils.configuration import config
    global _config
    _config = config


@pytest.fixture(autouse=True)
def before_method(mocker):
    import sys
    try:
        for name in sys.argv:
            if name.startswith("--"):
                sys.argv.remove(name)
                delattr(tf_flags.FLAGS, name)

    except AttributeError:
        pass

    for name in [PARAM_NAME, "s"] + [x[0] for x in PATCHED_CONFIGURATION_VALUES]:
        try:
            delattr(tf_flags.FLAGS, name)
        except:
            continue
    mocker.patch('sys.exit')
    global config, _config
    config = copy.deepcopy(_config)

    conf_with_defined_test_flags = {
        **configuration.cli_args_to_define,
        **dict.fromkeys([PARAM_NAME, 'dos', 'tres'], 'not_relevant')
    }
    mocker.patch.object(configuration, 'cli_args_to_define', conf_with_defined_test_flags)


def pass_cli_arg(cl_flags):
    import sys
    [sys.argv.append('--' + str(f[0]) + '=' + str(f[1])) for f in cl_flags]


def define_command_line_args(flag_names):
    [tf_flags.DEFINE_integer(flag, None, 'no-help :(') for flag in flag_names]


# noinspection PyProtectedMember
def set_file_param(name, value):
    config.file_defined_params.update({name: value})
    config._rebuild_full_config()


def set_model_param(name, value):
    config.update_model_params({name: value})


def set_launcher_param(name, value):
    config.update_launcher_params({name: value})


def assert_param(param_name, expected_value):
    def main(*args):
        config.update_tf_flags()
        assert config[param_name] == expected_value

    tf.compat.v1.app.run(main)


def test_should_return_none_when_no_args_passed():
    found_flag = config[PARAM_NAME]

    assert found_flag is None
    assert_param(PARAM_NAME, None)


def test_should_return_consts_param():
    set_file_param(PARAM_NAME, 45)

    found_flag = config[PARAM_NAME]

    assert found_flag == 45


def test_should_return_param_when_passed_as_cl_arg():
    define_command_line_args([PARAM_NAME])
    pass_cli_arg([(PARAM_NAME, 42)])
    assert_param(PARAM_NAME, 42)


def test_should_prioritize_cl_args_over_model_params():
    define_command_line_args([PARAM_NAME])
    pass_cli_arg([(PARAM_NAME, 42)])
    set_model_param(PARAM_NAME, 45)

    assert_param(PARAM_NAME, 42)


def test_should_prioritize_cl_args_over_file_params():
    define_command_line_args([PARAM_NAME])
    pass_cli_arg([(PARAM_NAME, 42)])
    set_file_param(PARAM_NAME, 45)

    assert_param(PARAM_NAME, 42)


def test_should_prioritize_model_params_over_file_params():
    set_model_param(PARAM_NAME, 42)
    set_file_param(PARAM_NAME, 45)

    assert_param(PARAM_NAME, 42)


def test_should_prioritize_launcher_params_over_model_params():
    set_launcher_param(PARAM_NAME, 42)
    set_model_param(PARAM_NAME, 45)
    set_file_param(PARAM_NAME, 48)

    assert_param(PARAM_NAME, 42)


def test_should_return_none_when_arg_not_defined():
    not_defined_arg_name = 'not_defined_arg_name'

    pass_cli_arg([(not_defined_arg_name, 42)])
    assert_param(not_defined_arg_name, None)


def test_should_log_when_flag_not_defined_during_training(mocker):
    launcher = DefaultLauncher([(FakeModel())])
    mocker.patch('src.utils.image_summaries.create_pair_summaries')
    mocker.patch('src.estimator.training.training.train')
    mocker.patch('src.estimator.launcher.providing_launcher.provide_launcher',
                 return_value=launcher)

    undefined_commandline_flag = ("foo", 34)
    defined_commandline_flag = (consts.BATCH_SIZE, 42)
    pass_cli_arg([undefined_commandline_flag, defined_commandline_flag])

    testing_utils.testing_helpers.run_app()

    log = list(filenames.get_run_text_logs_dir(launcher.runs_data[0]).iterdir())[0].read_text()
    lines = tuple(open(str(list(filenames.get_run_text_logs_dir(launcher.runs_data[0]).iterdir())[0]), 'r'))
    line_in_question = [x for x in lines if 'Undefined commandline flags' in x][0]
    assert "--foo=34" in line_in_question


def test_should_get_multiple_file_params():
    set_file_param('uno', 1)
    set_file_param('dos', 2)
    set_file_param('tres', 3)

    file_params = config.file_defined_params
    assert_that(file_params, has_entries({'uno': 1, 'dos': 2, 'tres': 3}))


def test_should_get_file_params():
    set_file_param('uno', 1)
    pass_cli_arg([(PARAM_NAME, 42)])

    def check(*args):
        params = config.file_defined_params
        assert_that(params, has_entry('uno', 1))
        assert_that(params, not_(has_entry(PARAM_NAME, 42)))

    tf.compat.v1.app.run(check)


def test_should_get_cli_args():
    define_command_line_args(['dos', 'tres'])
    set_file_param('uno', 1)
    pass_cli_arg([('dos', 2), ('tres', 3)])

    def check(*args):
        config.update_tf_flags()
        params = config.tf_flags
        assert_that(params, has_entries({'dos': 2, 'tres': 3}))
        assert_that(params, not_(has_entry('uno', 1)))

    tf.compat.v1.app.run(check)


def test_should_get_model_params():
    define_command_line_args(['uno'])
    pass_cli_arg([('uno', 2)])
    config.update_tf_flags()

    set_file_param('dos', 2)
    set_model_param('tres', 3)

    def check(*args):
        model_params = config.model_params
        assert_that(model_params, has_entries({'tres': 3}))
        assert_that(model_params, not_(has_entries({'uno': 1, 'dos': 2})))

    tf.compat.v1.app.run(check)


def test_check_defining_cli_args():
    if not tf_flags.FLAGS.find_module_defining_flag(consts.BATCH_SIZE):
        configuration.define_cli_args()
    pass_cli_arg(PATCHED_CONFIGURATION_VALUES)

    def main(*args):
        config.update_tf_flags()
        for flag in PATCHED_CONFIGURATION_VALUES:
            found_flag = config[flag[0]]
            if flag[0] != consts.EXCLUDED_KEYS:
                assert found_flag == flag[1]
            else:
                assert found_flag == ['1', '2', '3', 'foobar']

    tf.compat.v1.app.run(main)
    tf_flags.FLAGS.unparse_flags()
