from hamcrest import assert_that, contains

from estimator.training.integration.test_integration_training import FakeExperimentLauncher
from src.estimator.launcher.launchers import DefaultLauncher
from src.estimator.training import training
from src.utils import consts, filenames
from testing_utils import gen
from testing_utils.testing_classes import FakeModel


def prepare_mocks(launcher, mocker):
    mocker.patch('src.estimator.launcher.providing_launcher.provide_launcher', return_value=launcher)
    before_run_mock = mocker.patch('src.utils.before_run.prepare_env', autospec=True)
    mocker.patch('src.estimator.training.training.create_estimator', autospec=True)
    in_memory_training_mock = mocker.patch('src.estimator.training.training.in_memory_train_eval', autospec=True)
    return before_run_mock, in_memory_training_mock


def test_training_in_memory_only(mocker):
    launcher = DefaultLauncher([
        FakeModel()
    ])
    before_run_mock, in_memory_training_mock = prepare_mocks(launcher, mocker)

    distributed_training_mock = mocker.patch('src.estimator.training.training.distributed_train_eval', autospec=True)
    import sys

    training.main(sys.argv)

    before_run_mock.assert_called_once()
    in_memory_training_mock.assert_called_once()
    distributed_training_mock.assert_not_called()


def test_should_train_with_each_model(mocker):
    cnn_model_uno = FakeModel()
    cnn_model_dos = FakeModel()
    launcher = FakeExperimentLauncher([
        cnn_model_uno,
        cnn_model_dos
    ])

    before_run_mock, in_memory_training_mock = prepare_mocks(launcher, mocker)
    import sys

    training.main(sys.argv)

    assert before_run_mock.call_count == 2
    assert in_memory_training_mock.call_count == 2
    model_arguments = get_model_called_with(before_run_mock)
    assert_that(model_arguments,
                contains(*[x for x in launcher.runs_data]))


def get_model_called_with(before_run_mock):
    return [x[0][-1] for x in before_run_mock.call_args_list]


def test_should_pass_model_dir_to_estimator():
    model = FakeModel()
    run_data = gen.run_data(model)
    estimator = training.create_estimator(run_data)
    model_dir = estimator.params[consts.MODEL_DIR]
    assert model_dir == str(filenames.get_run_logs_data_dir(run_data))
